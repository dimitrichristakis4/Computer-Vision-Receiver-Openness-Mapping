#!/usr/bin/env python3
# Receiver openness tracking pipeline - Day 3 Play 2
#
# Processes a 3-segment clip: the last segment (end zone zoomed out) is annotated,
# the middle segment (end zone zoomed in) is skipped and copied raw.
#
# Coordinate system:
#   Y = depth past line of scrimmage, positive going downfield.
#       Derived from the horizontal pixel axis on the elevated sideline camera.
#       Y=0 is set at the QB's pixel position at snap. Downfield direction
#       is auto-detected by watching which way receivers move 0.5s post-snap.
#   X = lateral position, positive to the right.
#       X=0 at the OL cluster centroid in the end zone view.
#
# Receiver numbering: R1 is the receiver closest to the near sideline (highest
# pixel Y in the frame). No wide-side logic applied.

import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    import torch
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    warnings.warn("ultralytics not installed")

try:
    import easyocr as _easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    _easyocr = None

_ocr_reader = None
def get_ocr_reader():
    global _ocr_reader
    if not HAS_OCR: return None
    if _ocr_reader is None:
        _ocr_reader = _easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader

# PATHS
CV_DIR = Path(
    "/Users/dimitrichristakis/Desktop/AI Brain/In-depth/"
    "Sports Analytics-All/Sports Analytics-Queens Football-all/CV Test"
)
CLIP = CV_DIR / "Day 3 (03-22-2026):Team:Play 2.mov"
OUT  = CV_DIR / "Day3_annotated_v2.mp4"

FORMATION_RECV_COUNT = 4   # 3X1 -> 4 receivers

# YOLO
YOLO_MODEL      = "yolov8m.pt"
YOLO_CONF       = 0.20
YOLO_IOU        = 0.30   # lowered from 0.5 -> allow stacked players
YOLO_PERSON_CLS = 0
YOLO_BALL_CLS   = 32
BALL_CONF       = 0.20

# SNAP
STILLNESS_FRAMES = 52
STILLNESS_PX_THR = 5.0
SNAP_MOTION_MULT = 2.5
SNAP_LEAD_FRAMES = 10   # frames to subtract from detected snap: OL motion lags true snap

# THROW
THROW_MIN_OFFSET_S = 0.4
ARM_FLOW_MULT      = 3.5
LINEMEN_DROP_RATIO = 0.5

# OL POST-SNAP WINDOW
OL_STILLNESS_WINDOW_S = 0.5   # measure OL stillness in first 0.5s post-snap
TE_DEPTH_THR          = 80.0  # px downfield (horizontal pixel) over 1s: below = blocking TE
OL_LATERAL_MAX_GAP    = 150   # px: max cy gap to nearest OL neighbour; larger = lateral outlier (receiver)

# DEPTH CALIBRATION
# cx (horizontal pixel) = downfield (LOS -> end zone).  Elevated sideline captures
# this linearly - no perspective warp needed, just a single constant scale.
# Lateral (cy) uses build_lateral_cal() instead (perspective-corrected).
# Calibrate by measuring pixel gap between two visible yard lines in the frame.
DEPTH_PX_PER_YD = 45.0   # horizontal pixels per yard downfield (tune from yard lines)

# COACH FILTER
COACH_DISP_THR = 45

# DEFENSE TRACK FILTER
# A real defender is on the field the whole play (30+ frames).
# Leg-ghost tracks spawned by moving receiver limbs last only 2-5 frames.
# Any DEFENSE-labelled track shorter than this is downgraded to UNKNOWN.
MIN_DEFENSE_FRAMES = 12

# VISUAL (BGR)
COL = {
    "QB":      (0,   0,   220),
    "OFFENSE": (0,   200, 200),
    "DEFENSE": (220,  80,   0),
    "UNKNOWN": (140, 140, 140),
    "COACH":   (0,   140, 255),
    "OL":      (50,  200,  50),
}
SNAP_COL     = (0, 255,   0)
THROW_COL    = (0, 165, 255)
BALL_COL     = (0, 255, 255)
OOB_LINE_COL = (0, 255, 255)

# HSV
# QB red jersey (two hue ranges to catch full red spectrum)
HSV_QB_LO1   = np.array([0,   80,  70]);  HSV_QB_HI1   = np.array([10,  255, 255])
HSV_QB_LO2   = np.array([155,  80,  70]);  HSV_QB_HI2   = np.array([180, 255, 255])
# Offense yellow jersey
HSV_YEL_LO   = np.array([18,   80,  80]);  HSV_YEL_HI   = np.array([38,  255, 255])
# Defense/guardian-cap blue
HSV_BLU_LO   = np.array([85,   50,  40]);  HSV_BLU_HI   = np.array([135, 255, 255])
# QB gold helmet (no guardian cap) - richer/darker gold than jersey yellow
HSV_GOLD_LO  = np.array([18,  120,  80]);  HSV_GOLD_HI  = np.array([38,  255, 255])
HSV_GREEN_LO = np.array([35,   30,  30]);  HSV_GREEN_HI = np.array([85,  255, 255])
HSV_BRIGHT_LO = np.array([0,    0, 200]);  HSV_BRIGHT_HI = np.array([180, 60, 255])
HSV_SAT_LO    = np.array([0,  100, 100]);  HSV_SAT_HI    = np.array([180, 255, 255])



# HELPERS


def get_device():
    if HAS_YOLO and torch.backends.mps.is_available(): return "mps"
    if HAS_YOLO and torch.cuda.is_available():          return "cuda"
    return "cpu"


def classify_player(crop_bgr):
    """Returns (team, jcounts, shoe_score, qb_signal).
    team is OFFENSE / DEFENSE / UNKNOWN only - QB is a role, not a team.
    qb_signal=True when gold helmet + no guardian cap + red jersey detected."""
    empty = ("UNKNOWN", {"QB": 0, "OFFENSE": 0, "DEFENSE": 0}, 0.0, False)
    if crop_bgr is None or crop_bgr.size == 0: return empty
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 6: return empty

    # Zone 1: Helmet (top 15%)
    helmet_h = max(1, int(h * 0.15))
    helmet   = crop_bgr[:helmet_h, :]
    hsv_helm = cv2.cvtColor(helmet, cv2.COLOR_BGR2HSV)
    helm_px  = max(helmet.shape[0] * helmet.shape[1], 1)

    blue_helm = int(np.count_nonzero(cv2.inRange(hsv_helm, HSV_BLU_LO, HSV_BLU_HI)))
    gold_helm = int(np.count_nonzero(cv2.inRange(hsv_helm, HSV_GOLD_LO, HSV_GOLD_HI)))

    has_cap  = blue_helm / helm_px >= 0.10
    has_gold = gold_helm / helm_px >= 0.08

    # Zone 2: Jersey (20%–48%) - tightened to exclude blue pants
    j_top = int(h * 0.20)
    j_bot = int(h * 0.48)
    if j_bot <= j_top: j_bot = min(j_top + 4, h)
    jersey  = crop_bgr[j_top:j_bot, :]
    hsv_j   = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    jers_px = max(jersey.shape[0] * jersey.shape[1], 1)

    jcounts = {
        "QB":      int(np.count_nonzero(
                       cv2.inRange(hsv_j, HSV_QB_LO1, HSV_QB_HI1) |
                       cv2.inRange(hsv_j, HSV_QB_LO2, HSV_QB_HI2))),
        "OFFENSE": int(np.count_nonzero(cv2.inRange(hsv_j, HSV_YEL_LO, HSV_YEL_HI))),
        "DEFENSE": int(np.count_nonzero(cv2.inRange(hsv_j, HSV_BLU_LO, HSV_BLU_HI))),
    }

    qb_r  = jcounts["QB"]      / jers_px
    off_r = jcounts["OFFENSE"] / jers_px
    def_r = jcounts["DEFENSE"] / jers_px

    # Zone 3: Upper/lower body split check
    # Yellow upper body + blue lower body is unique to OFFENSE:
    #   offense = yellow jersey (upper) + blue pants (lower) -> clear colour split
    #   defense = blue jersey (upper)  + blue pants (lower) -> uniformly blue, no split
    # This catches split-box ghost detections: a lower-body crop of an offense player
    # reads blue pants -> classified DEFENSE, creating a phantom defender.
    u_top, u_bot = int(h * 0.20), int(h * 0.42)
    l_top, l_bot = int(h * 0.50), int(h * 0.68)
    offense_split = False
    if u_bot > u_top and l_bot > l_top and l_bot <= h:
        upper_body = crop_bgr[u_top:u_bot, :]
        lower_body = crop_bgr[l_top:l_bot, :]
        hsv_up = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
        hsv_lo = cv2.cvtColor(lower_body, cv2.COLOR_BGR2HSV)
        u_px   = max(upper_body.shape[0] * upper_body.shape[1], 1)
        l_px   = max(lower_body.shape[0] * lower_body.shape[1], 1)
        u_yel  = np.count_nonzero(cv2.inRange(hsv_up, HSV_YEL_LO, HSV_YEL_HI)) / u_px
        l_blu  = np.count_nonzero(cv2.inRange(hsv_lo, HSV_BLU_LO, HSV_BLU_HI)) / l_px
        offense_split = u_yel >= 0.20 and l_blu >= 0.12

    # QB signal: gold helmet, no guardian cap, red jersey
    qb_signal = has_gold and not has_cap and qb_r >= 0.02   # lowered from 0.04

    # Aspect ratio guard: a standing player is always notably taller than wide.
    # A partial-body crop (leg, pants fragment) is square or wider than tall.
    # Raise the DEFENSE confidence bar for near-square crops so a receiver's
    # extended leg or stray pants detection doesn't become a phantom defender.
    is_full_body = (h >= w * 1.3)
    def_thr = 0.04 if is_full_body else 0.18   # require much stronger blue signal on short/square crops
    # Height guard: crops shorter than ~50px cannot contain a full body.
    # Moving legs produce tall-narrow crops that pass the aspect-ratio check
    # but are still body-fragment detections - require very strong blue signal.
    if h < 50:
        def_thr = max(def_thr, 0.30)

    # Team (OFFENSE / DEFENSE / UNKNOWN only)
    # Split check takes highest priority - unambiguous kit signature.
    if offense_split:
        team = "OFFENSE"
    # Mixed-crop guard: two players in one bbox -> both colors strong -> UNKNOWN
    elif off_r >= 0.10 and def_r >= 0.08:
        team = "UNKNOWN"
    elif qb_signal or off_r >= 0.04:
        team = "OFFENSE"
    elif has_cap and def_r >= def_thr:
        team = "DEFENSE"
    elif def_r >= def_thr:
        team = "DEFENSE"
    else:
        # fallback: jersey plurality
        best = max(jcounts, key=jcounts.get)
        raw  = best if jcounts[best] / jers_px >= 0.05 else "UNKNOWN"
        team = "OFFENSE" if raw == "QB" else raw

    # Rescue: distant/small QB crops often land in DEFENSE because noisy blue
    # pixels cross the 0.04 threshold while the red jersey signal stays weak.
    # If red outweighs blue on the same crop, the player can't be a defender.
    # Real defenders have qb_r ≈ 0 so this never fires for them.
    if team == "DEFENSE" and qb_r >= 0.03 and qb_r >= def_r * 0.7:
        team = "OFFENSE"

    # Shoe/glove score (bottom 20%)
    shoe_score = 0.0
    lo = int(h * 0.8)
    if lo < h:
        lower    = crop_bgr[lo:, :]
        hsv_l    = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        bright   = cv2.inRange(hsv_l, HSV_BRIGHT_LO, HSV_BRIGHT_HI)
        sat      = cv2.inRange(hsv_l, HSV_SAT_LO,    HSV_SAT_HI)
        no_green = cv2.bitwise_not(cv2.inRange(hsv_l, HSV_GREEN_LO, HSV_GREEN_HI))
        shoe_score = float(np.count_nonzero(
            cv2.bitwise_or(bright, cv2.bitwise_and(sat, no_green))
        )) / max(lower.size // 3, 1)

    return team, jcounts, shoe_score, qb_signal


def detect_hard_cuts(video_path, total_frames):
    cap  = cv2.VideoCapture(str(video_path))
    diffs, prev = [], None
    for f in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (160, 90))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append((f, float(np.mean(
                np.abs(gray.astype(np.int16) - prev.astype(np.int16))))))
        prev = gray
    cap.release()
    if not diffs:
        t = total_frames // 3; return [t, 2*t]
    arr   = np.array([d for _, d in diffs])
    thr   = float(np.mean(arr)) + 3.0 * float(np.std(arr))
    cands = sorted([(f, d) for f, d in diffs if d > thr], key=lambda x: -x[1])
    sel   = []
    for f, _ in cands:
        if all(abs(f-s) >= 30 for s in sel): sel.append(f)
        if len(sel) >= 2: break
    sel.sort()
    if len(sel) < 2:
        t = total_frames // 3; sel = [t, 2*t]
    return sel


def detect_oob_line_green(video_path, start_frame, H, W):
    """Bottom edge of green field mask = near-sideline OOB line."""
    cap = cv2.VideoCapture(str(video_path))
    # Sample a few frames and average the green mask bottom edge
    y_candidates = []
    for offset in [5, 15, 30, 60]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
        ret, frame = cap.read()
        if not ret: continue
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, HSV_GREEN_LO, HSV_GREEN_HI)
        # Find the lowest row that has substantial green content
        row_sums = green.sum(axis=1)  # sum across columns
        threshold = W * 0.10 * 255    # at least 10% of row is green
        green_rows = np.where(row_sums > threshold)[0]
        if len(green_rows) > 0:
            y_candidates.append(int(green_rows[-1]))  # lowest green row
    cap.release()
    if not y_candidates:
        return None
    oob_y = int(np.median(y_candidates))
    # Add a small buffer below the field edge
    return min(oob_y + 20, H - 1)


def detect_field_landmarks(video_path, start_frame, oob_y, W):
    """
    Detect far-OOB line and both hash lines for lateral (X) yard calibration.
    CFL landmarks: near OOB=0 yds, near hash=24 yds, far hash=41 yds, far OOB=65 yds.
    Returns (far_oob_row, near_hash_row, far_hash_row).  Any value may be None if undetected.
    """
    cap          = cv2.VideoCapture(str(video_path))
    far_oob_cands = []
    hash_votes    = defaultdict(int)

    for offset in [5, 15, 30, 60, 100]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
        ret, frame = cap.read()
        if not ret: continue
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, HSV_GREEN_LO, HSV_GREEN_HI)

        row_g = green.sum(axis=1)
        g_rows = np.where(row_g > W * 0.08 * 255)[0]
        if len(g_rows) < 10: continue
        far_oob_cands.append(int(g_rows[0]))

        far_y  = int(g_rows[0])
        near_y = oob_y
        if near_y - far_y < 50: continue

        # White-line mask within field region (exclude green)
        white      = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        white_field = cv2.bitwise_and(white, cv2.bitwise_not(green))
        row_w = white_field.sum(axis=1).astype(float)
        row_w[:far_y]  = 0
        row_w[near_y:] = 0

        for r in range(far_y, near_y):
            if row_w[r] > W * 0.02 * 255:
                hash_votes[int(round(r / 8.0) * 8)] += 1

    cap.release()

    far_oob_row = int(np.median(far_oob_cands)) if far_oob_cands else None
    if far_oob_row is None: return None, None, None

    field_h = oob_y - far_oob_row
    if field_h < 50:       return far_oob_row, None, None

    strong = {r: v for r, v in hash_votes.items()
              if v >= 2 and far_oob_row < r < oob_y}
    if not strong:         return far_oob_row, None, None

    # Expected pixel rows for CFL hash marks (linear approximation)
    exp_near = oob_y - field_h * (24 / 65)   # 37% up from near OOB
    exp_far  = oob_y - field_h * (41 / 65)   # 63% up from near OOB

    candidates = sorted(strong.keys())
    near_hash = min(candidates, key=lambda r: abs(r - exp_near))
    far_cands  = [r for r in candidates if r < near_hash]
    far_hash   = min(far_cands, key=lambda r: abs(r - exp_far)) if far_cands else None
    return far_oob_row, near_hash, far_hash


def build_lateral_cal(oob_y, far_oob_row, near_hash_row, far_hash_row):
    """
    Returns a callable lateral_cal(pixel_y) → yards from near sideline (0–65).
    Uses all available landmark pairs; falls back gracefully to 2-point linear.
    """
    pts = [(oob_y, 0.0), (far_oob_row, 65.0)]
    if near_hash_row is not None: pts.append((near_hash_row, 24.0))
    if far_hash_row  is not None: pts.append((far_hash_row,  41.0))
    pts.sort(key=lambda p: p[0])                     # sort by pixel row ascending
    px_arr  = np.array([p[0] for p in pts], dtype=float)
    yds_arr = np.array([p[1] for p in pts], dtype=float)

    def lateral_cal(pixel_y):
        return float(np.interp(pixel_y, px_arr, yds_arr))
    return lateral_cal


def compute_clothing_hist(crop_bgr):
    """HSV H+S histogram of clothing zone (15%-80%), field-green pixels excluded."""
    if crop_bgr is None or crop_bgr.size == 0: return None
    h = crop_bgr.shape[0]
    y1, y2 = int(h * 0.15), int(h * 0.80)
    if y2 <= y1: return None
    zone = crop_bgr[y1:y2, :]
    hsv  = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(hsv, HSV_GREEN_LO, HSV_GREEN_HI))
    hist = cv2.calcHist([hsv], [0, 1], mask, [32, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def appearance_sim(t1, t2):
    """Returns 0.0 (completely different) to 1.0 (identical) across three signals."""
    score, weight = 0.0, 0.0

    # Clothing histogram - most informative (gloves, sleeves, pants color mix)
    h1, h2 = t1.get("hist"), t2.get("hist")
    if h1 is not None and h2 is not None:
        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        score += max(0.0, corr) * 0.5
        weight += 0.5

    # Body size (height + aspect ratio) - linemen vs skill players
    s1, s2 = t1.get("mean_size"), t2.get("mean_size")
    if s1 and s2:
        h_diff = abs(s1[0] - s2[0]) / max(s1[0], s2[0], 1)
        a_diff = abs(s1[1] - s2[1]) / max(abs(s1[1]) + abs(s2[1]), 0.01) * 2
        score += max(0.0, 1.0 - h_diff - a_diff) * 0.3
        weight += 0.3

    # Jersey number - if readable, unique identifier
    n1, n2 = t1.get("jersey_number"), t2.get("jersey_number")
    if n1 and n2:
        score += (1.0 if n1 == n2 else 0.0) * 0.2
        weight += 0.2

    return score / weight if weight > 0 else 0.5


def run_jersey_ocr(video_path, tracks, start_frame, end_frame, samples=6):
    """Sample frames and OCR jersey numbers. Stores jersey_number in each track."""
    reader = get_ocr_reader()
    if reader is None: return
    print("  Running jersey OCR...")
    sample_frames = np.linspace(start_frame, end_frame, samples + 2,
                                dtype=int)[1:-1].tolist()
    number_votes = defaultdict(lambda: defaultdict(int))
    cap = cv2.VideoCapture(str(video_path))
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        for tid, track in tracks.items():
            if fi not in track["bboxes"]: continue
            x1, y1, x2, y2 = track["bboxes"][fi]
            bh, bw = y2 - y1, x2 - x1
            # Crop jersey zone: 20%-55% height, centre 70% width
            jy1 = y1 + int(bh * 0.20);  jy2 = y1 + int(bh * 0.55)
            jx1 = x1 + int(bw * 0.15);  jx2 = x2 - int(bw * 0.15)
            if jy2 <= jy1 or jx2 <= jx1: continue
            crop = frame[jy1:jy2, jx1:jx2]
            if crop.size == 0: continue
            crop_up = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4),
                                 interpolation=cv2.INTER_CUBIC)
            try:
                results = reader.readtext(crop_up, allowlist='0123456789',
                                          detail=1, min_size=8)
            except Exception:
                continue
            for (_, text, conf) in results:
                text = text.strip()
                if text.isdigit() and 1 <= int(text) <= 99 and conf > 0.45:
                    number_votes[tid][text] += 1
    cap.release()
    for tid, votes in number_votes.items():
        if votes:
            tracks[tid]["jersey_number"] = max(votes, key=votes.get)
    found = sum(1 for t in tracks.values() if t.get("jersey_number"))
    print(f"  Jersey numbers read on {found}/{len(tracks)} tracks")


def track_segment(video_path, start_frame, end_frame, model, device,
                  oob_y=None, far_oob_row=None, seed_conf=0.15, seed_frames=90):
    """YOLO+ByteTrack. Returns (tracks, frame_dets).
    First seed_frames frames use seed_conf (lower) to catch distant/faint players.
    Players above far_oob_row or below oob_y are marked OOB and excluded."""
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            for tr in model.predictor.trackers: tr.reset()

    tracks       = {}
    color_votes  = defaultdict(lambda: defaultdict(int))
    shoe_votes   = defaultdict(list)
    qb_sig_votes = defaultdict(int)
    hist_accum   = defaultdict(list)
    size_accum   = defaultdict(list)
    frame_dets   = {}

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fi = start_frame
    while fi <= end_frame:
        ret, frame = cap.read()
        if not ret: break
        H_f, W_f = frame.shape[:2]
        conf_this = seed_conf if (fi - start_frame) < seed_frames else YOLO_CONF
        results = model.track(frame, persist=True, classes=[YOLO_PERSON_CLS],
                               conf=conf_this, iou=YOLO_IOU,
                               tracker=str(CV_DIR / "bytetrack_sticky.yaml"),
                               device=device, verbose=False)
        dets = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            ids  = results[0].boxes.id.cpu().numpy().astype(int)
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                x1i,y1i = max(0,int(x1)), max(0,int(y1))
                x2i,y2i = min(W_f-1,int(x2)), min(H_f-1,int(y2))
                if x2i<=x1i or y2i<=y1i: continue
                crop = frame[y1i:y2i, x1i:x2i]
                team, jc, shoe, qb_sig = classify_player(crop)
                color_votes[int(tid)][team] += 1
                shoe_votes[int(tid)].append(shoe)
                if qb_sig: qb_sig_votes[int(tid)] += 1
                hist = compute_clothing_hist(crop)
                if hist is not None: hist_accum[int(tid)].append(hist)
                bh, bw = y2i - y1i, x2i - x1i
                if bh > 0: size_accum[int(tid)].append((bh, bw / bh))
                cx, cy = (x1i+x2i)/2, (y1i+y2i)/2
                is_oob = (oob_y is not None and cy > oob_y) or \
                         (far_oob_row is not None and cy < far_oob_row - 10)
                tid_i = int(tid)
                if tid_i not in tracks:
                    tracks[tid_i] = {"id":tid_i,"team":"UNKNOWN",
                                     "positions":{},"bboxes":{},"shoe_score":0.0}
                if not is_oob:
                    tracks[tid_i]["positions"][fi] = (cx, cy)
                    tracks[tid_i]["bboxes"][fi]    = (x1i,y1i,x2i,y2i)
                dets.append((tid_i,x1i,y1i,x2i,y2i,team,jc,shoe,is_oob,qb_sig))
        frame_dets[fi] = dets
        fi += 1
        if (fi - start_frame) % 200 == 0:
            print(f"    frame {fi}/{end_frame}")
    cap.release()
    for tid, votes in color_votes.items():
        non_unk = {k:v for k,v in votes.items() if k!="UNKNOWN"}
        if non_unk:
            total    = sum(non_unk.values())
            def_frac = non_unk.get("DEFENSE", 0) / total
            # Lock as DEFENSE if it dominates (≥70%) - prevents a few noisy
            # OFFENSE frames (e.g. from a split-box upper crop) from flipping
            # a confirmed defender to OFFENSE via bare majority.
            if def_frac >= 0.70:
                tracks[tid]["team"] = "DEFENSE"
            else:
                tracks[tid]["team"] = max(non_unk, key=non_unk.get)
        tracks[tid]["qb_signals"] = qb_sig_votes.get(tid, 0)
        if shoe_votes[tid]: tracks[tid]["shoe_score"] = float(np.mean(shoe_votes[tid]))
        if hist_accum[tid]:
            tracks[tid]["hist"] = np.mean(hist_accum[tid], axis=0).astype(np.float32)
        if size_accum[tid]:
            heights  = [s[0] for s in size_accum[tid]]
            aspects  = [s[1] for s in size_accum[tid]]
            tracks[tid]["mean_size"] = (float(np.mean(heights)),
                                        float(np.mean(aspects)))
    # Suppress DEFENSE labels on short-lived tracks.
    # Moving receiver legs trigger spurious 2-5 frame DEFENSE tracks.
    # Real defenders persist across the whole play.
    for tid, track in tracks.items():
        if (track["team"] == "DEFENSE"
                and len(track.get("positions", {})) < MIN_DEFENSE_FRAMES):
            track["team"] = "UNKNOWN"

    # Centroid proximity suppression: a receiver's own leg fragment will have
    # its centroid nearly on top of the receiver's centroid (same body).
    # A real defender - even in press coverage - is a separate person and
    # always sits 40+ px away.  If a DEFENSE track centroid is within
    # GHOST_CENTROID_PX of any OFFENSE centroid in >40% of shared frames,
    # it is a limb ghost, not a real defender.
    GHOST_CENTROID_PX = 30
    off_positions = {tid: t["positions"] for tid, t in tracks.items()
                     if t["team"] == "OFFENSE"}
    for tid, track in tracks.items():
        if track["team"] != "DEFENSE": continue
        positions = track.get("positions", {})
        if not positions: continue
        ghost_frames = 0
        shared_frames = 0
        for fi, (dcx, dcy) in positions.items():
            for opos in off_positions.values():
                if fi not in opos: continue
                shared_frames += 1
                ocx, ocy = opos[fi]
                if math.hypot(dcx - ocx, dcy - ocy) < GHOST_CENTROID_PX:
                    ghost_frames += 1
                break
        if shared_frames > 0 and ghost_frames / shared_frames > 0.40:
            track["team"] = "UNKNOWN"

    return tracks, frame_dets


def detect_ball_post_snap(video_path, snap_frame, end_frame, model, device):
    if snap_frame is None: return {}
    ball_dets = {}
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
    for fi in range(snap_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret: break
        res = model.predict(frame, classes=[YOLO_BALL_CLS], conf=BALL_CONF,
                            iou=0.3, device=device, verbose=False)
        if res and res[0].boxes is not None and len(res[0].boxes):
            for (bx1,by1,bx2,by2), bc in zip(
                    res[0].boxes.xyxy.cpu().numpy(),
                    res[0].boxes.conf.cpu().numpy()):
                if (bx2-bx1) < 100 and (by2-by1) < 100:
                    ball_dets.setdefault(fi,[]).append(
                        (int(bx1),int(by1),int(bx2),int(by2),float(bc)))
    cap.release()
    return ball_dets


# SNAP

def detect_snap_ol_stillness(tracks, ol_ids, start_frame, end_frame, fps):
    ol_tracks = [t for t in tracks.values() if t["id"] in ol_ids]
    if not ol_tracks: return None
    frames = sorted({fr for t in ol_tracks for fr in t["positions"]
                     if start_frame <= fr <= end_frame})
    if len(frames) < STILLNESS_FRAMES + 10: return None
    motion, prev_pos = [], {}
    for fr in frames:
        mags = []
        for t in ol_tracks:
            if fr in t["positions"]:
                cur = t["positions"][fr]
                if t["id"] in prev_pos:
                    mags.append(math.hypot(cur[0]-prev_pos[t["id"]][0],
                                           cur[1]-prev_pos[t["id"]][1]))
                prev_pos[t["id"]] = cur
        if mags: motion.append((fr, float(np.mean(mags))))
    if len(motion) < STILLNESS_FRAMES + 5: return None
    arr = np.array([m for _,m in motion])
    best_end, best_mean = None, float("inf")
    run_start = None
    for i,(_, m) in enumerate(motion):
        if m < STILLNESS_PX_THR:
            if run_start is None: run_start = i
            if (i-run_start+1) >= STILLNESS_FRAMES:
                wm = float(np.mean(arr[run_start:i+1]))
                if wm < best_mean: best_mean, best_end = wm, i
        else:
            run_start = None
    if best_end is None:
        for i in range(len(arr)-STILLNESS_FRAMES):
            wm = float(np.mean(arr[i:i+STILLNESS_FRAMES]))
            if wm < best_mean: best_mean, best_end = wm, i+STILLNESS_FRAMES-1
    if best_end is None: return None
    threshold = max(STILLNESS_PX_THR*2, best_mean*SNAP_MOTION_MULT)
    for fr,m in motion[best_end+1:]:
        if m > threshold: return fr
    return None


# OL IDENTIFICATION (post-snap stillness)

def identify_ol_post_snap(tracks, snap_frame, fps):
    """5 yellow tracks with lowest displacement in first 0.5s post-snap = OL.
    Lateral outliers (split-wide receivers with low motion) are purged afterward:
    an OL candidate whose cy at snap is more than OL_LATERAL_MAX_GAP pixels from
    its nearest neighbour in the candidate set is a receiver, not a lineman."""
    if snap_frame is None: return set()
    window_end = snap_frame + int(fps * OL_STILLNESS_WINDOW_S)
    yellow = [t for t in tracks.values() if t["team"] == "OFFENSE"]
    displacements = {}
    for t in yellow:
        frames_in_window = sorted(
            fr for fr in t["positions"] if snap_frame <= fr <= window_end)
        if len(frames_in_window) < 2:
            displacements[t["id"]] = 0.0
            continue
        p_first = t["positions"][frames_in_window[0]]
        p_last  = t["positions"][frames_in_window[-1]]
        displacements[t["id"]] = math.hypot(
            p_last[0]-p_first[0], p_last[1]-p_first[1])
    if not displacements: return set()
    sorted_ids = sorted(displacements, key=displacements.get)
    candidates = list(sorted_ids[:5])

    # Purge lateral outliers from OL candidates.
    # Real OL are shoulder-to-shoulder; a wide receiver mistakenly included will
    # have a cy gap >> OL_LATERAL_MAX_GAP to its nearest OL neighbour.
    snap_cys = {}
    for tid in candidates:
        t = tracks.get(tid)
        if t is None or not t["positions"]: continue
        nf = min(t["positions"], key=lambda f: abs(f - snap_frame))
        if abs(nf - snap_frame) <= 30:
            snap_cys[tid] = t["positions"][nf][1]

    if len(snap_cys) >= 3:
        by_cy = sorted(snap_cys.items(), key=lambda x: x[1])
        cy_vals = [cy for _, cy in by_cy]
        final = []
        for i, (tid, cy) in enumerate(by_cy):
            lo_gap = abs(cy - cy_vals[i-1]) if i > 0 else float("inf")
            hi_gap = abs(cy - cy_vals[i+1]) if i < len(cy_vals)-1 else float("inf")
            nearest_gap = min(lo_gap, hi_gap)
            if nearest_gap <= OL_LATERAL_MAX_GAP:
                final.append(tid)
            else:
                print(f"  OL outlier purged: #{tid} cy={cy:.0f}px "
                      f"(nearest OL neighbour gap={nearest_gap:.0f}px)")
        return set(final)

    return set(candidates)


# TE IDENTIFICATION

def identify_te(tracks, ol_ids, snap_frame, fps):
    """
    The most-still non-OL OFFENSE player post-snap is the TE candidate.
    Displacement < TE_BLOCK_DISP_THR → blocking TE (treat as OL, same as 3X1).
    Displacement >= threshold         → receiving TE (3X1 S, count as receiver).
    Returns (te_id, is_blocking).  te_id=None if no non-OL OFFENSE players found.
    """
    if snap_frame is None: return None, False
    window_end = snap_frame + int(fps * OL_STILLNESS_WINDOW_S)
    yellow_non_ol = [t for t in tracks.values()
                     if t["team"] == "OFFENSE" and t["id"] not in ol_ids]
    displacements = {}
    for t in yellow_non_ol:
        frames_w = sorted(fr for fr in t["positions"]
                          if snap_frame <= fr <= window_end)
        if len(frames_w) < 2:
            displacements[t["id"]] = 999.0
            continue
        p0, p1 = t["positions"][frames_w[0]], t["positions"][frames_w[-1]]
        displacements[t["id"]] = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
    if not displacements: return None, False
    te_id       = min(displacements, key=displacements.get)
    is_blocking = displacements[te_id] < TE_BLOCK_DISP_THR
    return te_id, is_blocking


# QB IDENTIFICATION

def find_qb_sideline(tracks, snap_frame, ol_ids):
    """OFFENSE player with most gold-helmet (qb_signal) frames closest to OL centroid."""
    if snap_frame is None: return None
    offense = [t for t in tracks.values()
               if t["team"] == "OFFENSE" and t["id"] not in ol_ids
               and t.get("qb_signals", 0) > 0]
    if not offense:
        # fallback: any non-OL OFFENSE track - pick closest to OL centroid
        offense = [t for t in tracks.values()
                   if t["team"] == "OFFENSE" and t["id"] not in ol_ids]
    if not offense: return None

    # OL centroid - use generous frame tolerance (45) so a brief detection gap
    # around snap_frame doesn't wipe the reference
    ol_positions = []
    for t in tracks.values():
        if t["id"] not in ol_ids or not t["positions"]: continue
        nf = min(t["positions"], key=lambda f: abs(f - snap_frame))
        if abs(nf - snap_frame) <= 45:
            ol_positions.append(t["positions"][nf])
    if not ol_positions:
        # last resort: use all OL positions ever seen
        for t in tracks.values():
            if t["id"] not in ol_ids or not t["positions"]: continue
            nf = min(t["positions"], key=lambda f: abs(f - snap_frame))
            ol_positions.append(t["positions"][nf])
    if not ol_positions: return None

    ol_cx = float(np.mean([p[0] for p in ol_positions]))
    ol_cy = float(np.mean([p[1] for p in ol_positions]))

    def score(t):
        if not t["positions"]: return (0, 1e9)
        nf = min(t["positions"], key=lambda f: abs(f - snap_frame))
        d  = math.hypot(t["positions"][nf][0] - ol_cx,
                        t["positions"][nf][1] - ol_cy)
        return (-t.get("qb_signals", 0), d)   # most signals first, then closest

    return min(offense, key=score)


def find_qb_endzone(tracks, snap_approx):
    """QB-colored player surrounded by most yellow players within 200px."""
    qb_cands = [t for t in tracks.values() if t["team"] == "QB"]
    yellow   = [t for t in tracks.values() if t["team"] == "OFFENSE"]
    if not qb_cands: return None
    if not yellow:   return qb_cands[0]
    best_qb, best_count = None, -1
    for qb in qb_cands:
        if not qb["positions"]: continue
        nf     = min(qb["positions"], key=lambda f: abs(f-snap_approx))
        qb_pos = qb["positions"][nf]
        count  = 0
        for yt in yellow:
            if not yt["positions"]: continue
            nf2 = min(yt["positions"], key=lambda f: abs(f-snap_approx))
            yp  = yt["positions"][nf2]
            if math.hypot(yp[0]-qb_pos[0], yp[1]-qb_pos[1]) < 200:
                count += 1
        if count > best_count: best_count, best_qb = count, qb
    return best_qb


# TRACK CONSOLIDATION

def consolidate_offense_tracks(tracks, end_frame, fps,
                                max_gap_s=1.00, max_dist_px=200):
    """
    Physically merge fragmented OFFENSE track chains so downstream movement
    checks operate on a complete positional history, not a short fragment.

    For each OFFENSE track that ends before end_frame: find the best-matching
    OFFENSE track that starts within max_gap_s at ≤ max_dist_px, score it on
    spatial proximity + appearance similarity, and if the score is < 0.75 merge
    the continuation's data into the anchor.

    Returns (tracks, merge_map) where merge_map = {secondary_id: primary_id}.
    The render loop uses merge_map to find the role/team of a detection whose
    track was absorbed into another.
    """
    max_gap   = int(fps * max_gap_s)
    merge_map = {}
    changed   = True

    while changed:
        changed = False
        offense = [(tid, t) for tid, t in list(tracks.items())
                   if t["team"] == "OFFENSE" and t["positions"]]
        offense.sort(key=lambda x: min(x[1]["positions"]))  # process earliest first

        for tid_a, ta in offense:
            if tid_a not in tracks: continue
            last_f  = max(ta["positions"])
            last_p  = ta["positions"][last_f]
            if last_f >= end_frame: continue

            best_b, best_score = None, float("inf")
            for tid_b, tb in offense:
                if tid_b == tid_a or tid_b not in tracks: continue
                if not tb["positions"]: continue
                first_f = min(tb["positions"])
                if first_f <= last_f or first_f > last_f + max_gap: continue
                d = math.hypot(
                    tb["positions"][first_f][0] - last_p[0],
                    tb["positions"][first_f][1] - last_p[1])
                if d > max_dist_px: continue
                norm_d = d / max_dist_px
                sim    = appearance_sim(ta, tb)
                score  = 0.4 * norm_d + 0.6 * (1.0 - sim)
                if score < best_score:
                    best_score, best_b = score, tid_b

            if best_b is not None and best_score < 0.75:
                # Absorb best_b into tid_a
                tracks[tid_a]["positions"].update(tracks[best_b]["positions"])
                tracks[tid_a]["bboxes"].update(tracks[best_b].get("bboxes", {}))
                tracks[tid_a]["qb_signals"] = (
                    tracks[tid_a].get("qb_signals", 0) +
                    tracks[best_b].get("qb_signals", 0))
                if not tracks[tid_a].get("jersey_number"):
                    tracks[tid_a]["jersey_number"] = tracks[best_b].get("jersey_number")
                # Record merge; redirect any prior merge_map entries pointing at best_b
                merge_map[best_b] = tid_a
                for k in list(merge_map):
                    if merge_map[k] == best_b:
                        merge_map[k] = tid_a
                del tracks[best_b]
                changed = True
                break   # restart while loop after each merge

    return tracks, merge_map


# RECEIVERS

def stitch_tracks(tracks, id_to_label, protected_ids, end_frame,
                  max_gap_frames=14, max_dist_px=150,
                  search_teams=("OFFENSE",)):
    """
    Forward-stitch broken tracks by proximity.
    When a labeled track ends before end_frame, find the closest unlabeled
    track (of the allowed teams) that starts within max_gap_frames and
    max_dist_px of the last known position, and inherit the label.
    Iterates until stable to handle chains of fragmentation.
    """
    labeled = dict(id_to_label)
    changed = True
    while changed:
        changed = False
        currently_labeled = set(labeled)
        for tid, label in list(labeled.items()):
            t = tracks.get(tid)
            if t is None or not t["positions"]: continue
            last_frame = max(t["positions"])
            if last_frame >= end_frame: continue
            last_pos = t["positions"][last_frame]
            best_tid, best_score = None, float("inf")
            for cid, ct in tracks.items():
                if cid in currently_labeled:       continue
                if ct["team"] not in search_teams: continue
                if cid in protected_ids:           continue
                if not ct["positions"]:            continue
                first_frame = min(ct["positions"])
                if first_frame <= last_frame:                   continue
                if first_frame > last_frame + max_gap_frames:  continue
                first_pos = ct["positions"][first_frame]
                d = math.hypot(first_pos[0] - last_pos[0],
                               first_pos[1] - last_pos[1])
                if d > max_dist_px: continue
                # Combined: 40% spatial proximity + 60% appearance dissimilarity
                norm_d   = d / max_dist_px
                sim      = appearance_sim(t, ct)
                combined = 0.4 * norm_d + 0.6 * (1.0 - sim)
                if combined < best_score:
                    best_score, best_tid = combined, cid
            # Accept if combined score is below threshold (0.75 = generous gate)
            if best_tid is not None and best_score < 0.75:
                labeled[best_tid] = label
                currently_labeled.add(best_tid)
                changed = True
    return labeled


def number_receivers(tracks, qb_track, snap_frame, ol_ids, n_recv, fps):
    """R1 = highest Y pixel (nearest sideline). Excludes OL ids and QB.
    Blocking TEs are caught after counting by evict_blocking_receivers."""
    if qb_track is None or snap_frame is None: return {}
    yellow = [t for t in tracks.values() if t["team"] == "OFFENSE"]

    def pos_near(t, fr, tol=45):
        if fr in t["positions"]: return t["positions"][fr]
        if not t["positions"]:   return None
        nf = min(t["positions"], key=lambda f: abs(f-fr))
        return t["positions"][nf] if abs(nf-fr) <= tol else None

    # Players must have a real YOLO detection within ~0.6s of snap.
    # This excludes pre-snap players who left before snap (ghost receivers).
    # Synthetic tracks are exempt - their positions are constructed at snap time.
    snap_presence_tol = max(45, int(fps * 0.8))
    def has_snap_presence(t):
        if t.get("synthetic"):
            return True
        bboxes = t.get("bboxes", {})
        if not bboxes:
            return False
        nf = min(bboxes, key=lambda f: abs(f - snap_frame))
        return abs(nf - snap_frame) <= snap_presence_tol

    # OL lateral band at snap. Stray OL who escaped identify_ol_post_snap will
    # have their snap cy inside this band; real receivers are split wide and sit
    # outside it. Excluding band-internal candidates kills the "ghost R2 = OL"
    # case without relying on post-snap displacement.
    ol_cys_at_snap = []
    for tid in ol_ids:
        t = tracks.get(tid)
        if t is None or not t["positions"]: continue
        nf = min(t["positions"], key=lambda f: abs(f - snap_frame))
        if abs(nf - snap_frame) <= 30:
            ol_cys_at_snap.append(t["positions"][nf][1])
    ol_cy_min = ol_cy_max = None
    if len(ol_cys_at_snap) >= 3:
        ol_cy_min, ol_cy_max = min(ol_cys_at_snap), max(ol_cys_at_snap)

    def in_ol_band(p):
        if ol_cy_min is None: return False
        return ol_cy_min <= p[1] <= ol_cy_max

    # Backfield exclusion: RBs/FBs cluster within ~5 yards of the QB at snap.
    # Pure Euclidean distance - no OL-band check needed because any legitimate
    # receiver (slot or wide) is split at least 6+ yards from the QB at the LOS.
    BACKFIELD_DIST_PX = 240   # ~5.3 yards at DEPTH_PX_PER_YD=45
    qb_pos_snap = pos_near(qb_track, snap_frame)

    def is_backfield(p):
        if qb_pos_snap is None: return False
        return math.hypot(p[0] - qb_pos_snap[0], p[1] - qb_pos_snap[1]) <= BACKFIELD_DIST_PX

    with_pos = [(t, pos_near(t, snap_frame)) for t in yellow]
    with_pos = [(t, p) for t, p in with_pos if p is not None]
    non_ol   = [(t, p) for t, p in with_pos
                if t["id"] not in ol_ids
                and t["id"] != qb_track["id"]
                and has_snap_presence(t)
                and not in_ol_band(p)
                and not is_backfield(p)]
    # Real tracks first, synthetic (supplemental scan) tracks last within each group.
    # Prevents a spurious synthetic injection from stealing a slot from a real receiver.
    non_ol.sort(key=lambda x: (1 if x[0].get("synthetic") else 0, -x[1][1]))
    print(f"  number_receivers: {len(non_ol)} eligible non-OL players "
          f"(OL band cy=[{ol_cy_min},{ol_cy_max}])")
    for i, (t, p) in enumerate(non_ol[:n_recv + 2]):
        print(f"    slot {i+1}: id={t['id']} synthetic={t.get('synthetic',False)} "
              f"cy={p[1]:.0f} bboxes={len(t.get('bboxes',{}))} positions={len(t['positions'])}")
    return {f"R{i+1}": t["id"] for i, (t, _) in enumerate(non_ol[:n_recv])}


# RETROACTIVE BLOCKER EVICTION

def evict_blocking_receivers(receiver_map, tracks, snap_frame, fps):
    """
    Check every counted receiver's post-snap downfield movement over a 1s window.
    In the elevated sideline view:
      p[0] = horizontal pixel = DOWNFIELD (field Y)
      p[1] = vertical pixel   = LATERAL   (field X, sideline to sideline)
    Receivers MUST move downfield to run a route. Blocking TEs stay at LOS.
    If downfield displacement < TE_DEPTH_THR the player is a blocker - evict.
    Returns (updated_receiver_map, evicted_track_ids).
    """
    if not receiver_map or snap_frame is None:
        return receiver_map, set()

    window_end = snap_frame + int(fps * 1.0)
    evicted    = set()

    for _, tid in receiver_map.items():
        t = tracks.get(tid)
        if t is None or not t["positions"]:
            evicted.add(tid)   # merged away - reassign slot
            continue
        frames_w = sorted(fr for fr in t["positions"]
                          if snap_frame <= fr <= window_end)
        if len(frames_w) < 2:
            continue   # insufficient post-snap data - don't assume blocker
        p0, p1 = t["positions"][frames_w[0]], t["positions"][frames_w[-1]]
        d_downfield = abs(p1[0] - p0[0])   # horizontal pixel = DOWNFIELD (field Y)
        # Lateral movement (p[1]) is intentionally NOT checked - blocking TEs
        # shuffle sideways and would falsely pass a lateral-only check.
        if d_downfield < TE_DEPTH_THR:
            evicted.add(tid)

    if not evicted:
        return receiver_map, set()

    # Remove evicted and renumber preserving relative order
    kept    = [tid for _, tid in sorted(receiver_map.items(),
               key=lambda x: int(x[0][1:])) if tid not in evicted]
    updated = {f"R{i+1}": tid for i, tid in enumerate(kept)}
    print(f"  Evicted as blockers: {evicted} → receivers now {updated}")
    return updated, evicted


# THROW DETECTION

def detect_throw_arm_motion(video_path, qb_track, snap_frame, end_frame, fps,
                            y_sign=1):
    if qb_track is None or not qb_track["bboxes"]: return None
    min_fr = snap_frame + int(fps * THROW_MIN_OFFSET_S)
    cap    = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_fr)
    prev_gray, flow_series = None, []
    for fr in range(min_fr, end_frame+1):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = qb_track["bboxes"].get(fr)
        if bbox is None and qb_track["bboxes"]:
            nf   = min(qb_track["bboxes"], key=lambda f: abs(f-fr))
            bbox = qb_track["bboxes"][nf] if abs(nf-fr)<=8 else None
        if bbox is not None and prev_gray is not None:
            x1,y1,x2,y2 = bbox
            mid_y = y1 + (y2-y1)//2
            if x2>x1 and mid_y>y1:
                pp = prev_gray[y1:mid_y, x1:x2]
                pc = gray     [y1:mid_y, x1:x2]
                if pp.shape==pc.shape and pp.size>=64:
                    flow = cv2.calcOpticalFlowFarneback(
                        pp, pc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow_series.append(
                        (fr, float(np.mean(np.linalg.norm(flow, axis=2)))))
        prev_gray = gray
    cap.release()
    if len(flow_series) < 5: return None
    arr  = np.array([m for _,m in flow_series])
    base = float(np.mean(arr[:max(5,len(arr)//4)]))
    std  = float(np.std (arr[:max(5,len(arr)//4)])) + 1e-6
    thr  = base + ARM_FLOW_MULT * std
    # Return the PEAK frame of the first spike, not the onset.
    # Onset = arm going back (wind-up). Peak = maximum velocity = release.
    in_spike, best_fr, best_m = False, None, -1
    for fr, m in flow_series:
        if m > thr:
            in_spike = True
            if m > best_m:
                best_m, best_fr = m, fr
        elif in_spike:
            break   # spike ended - return its peak
    return best_fr


def detect_throw_ball_departure(ball_dets, qb_track, snap_frame, fps):
    if not ball_dets or qb_track is None or not qb_track["bboxes"]: return None
    min_fr = snap_frame + int(fps * THROW_MIN_OFFSET_S)
    for fr in sorted(ball_dets):
        if fr < min_fr: continue
        bbox = qb_track["bboxes"].get(fr)
        if bbox is None and qb_track["bboxes"]:
            nf   = min(qb_track["bboxes"], key=lambda f: abs(f-fr))
            bbox = qb_track["bboxes"][nf] if abs(nf-fr)<=5 else None
        if bbox is None: continue
        qx1,qy1,qx2,qy2 = bbox
        for (bx1,by1,bx2,by2,_) in ball_dets[fr]:
            bcx,bcy = (bx1+bx2)/2,(by1+by2)/2
            if not (qx1-30<=bcx<=qx2+30 and qy1-30<=bcy<=qy2+30):
                return fr
    return None


def detect_throw_linemen_drop(tracks, ol_ids, snap_frame, end_frame, fps):
    min_fr = snap_frame + int(fps * THROW_MIN_OFFSET_S)
    rel    = [t for t in tracks.values()
              if t["id"] in ol_ids or t["team"]=="DEFENSE"]
    frames = sorted({fr for t in rel for fr in t["positions"]
                     if min_fr<=fr<=end_frame})
    if len(frames) < 15: return None
    motion, prev_pos = [], {}
    for fr in frames:
        mags = []
        for t in rel:
            if fr in t["positions"]:
                cur = t["positions"][fr]
                if t["id"] in prev_pos:
                    mags.append(math.hypot(cur[0]-prev_pos[t["id"]][0],
                                           cur[1]-prev_pos[t["id"]][1]))
                prev_pos[t["id"]] = cur
        if mags: motion.append((fr, float(np.mean(mags))))
    arr = np.array([m for _,m in motion])
    for i in range(8, len(motion)):
        fr,m    = motion[i]
        avg_pre = float(np.mean(arr[max(0,i-8):i]))
        if avg_pre > 8 and m < avg_pre * LINEMEN_DROP_RATIO: return fr
    return None


def filter_coaches(tracks, snap_approx, throw_approx):
    end_ref = throw_approx if throw_approx else snap_approx + 120
    for t in tracks.values():
        if t["team"] != "DEFENSE" or not t["positions"]: continue
        after = [fr for fr in t["positions"] if fr >= snap_approx]
        if len(after) < 2: continue
        p1 = t["positions"][after[0]]
        p2 = t["positions"][min(t["positions"], key=lambda f: abs(f-end_ref))]
        if math.hypot(p2[0]-p1[0], p2[1]-p1[1]) < COACH_DISP_THR:
            t["team"] = "COACH"


# COORDINATE CALIBRATION

def detect_y_direction(tracks, snap_frame, fps):
    """Return +1 or -1: which horizontal direction is positive Y (downfield).
    Measure net horizontal displacement of non-OL yellow tracks 0.5s post-snap."""
    window_end = snap_frame + int(fps * 0.5)
    yellow = [t for t in tracks.values() if t["team"] == "OFFENSE"]
    deltas = []
    for t in yellow:
        frames_before = [fr for fr in t["positions"] if snap_frame <= fr <= window_end]
        if len(frames_before) < 2: continue
        p_first = t["positions"][frames_before[0]]
        p_last  = t["positions"][frames_before[-1]]
        deltas.append(p_last[0] - p_first[0])  # horizontal pixel change
    if not deltas: return 1
    return 1 if float(np.mean(deltas)) >= 0 else -1


def get_y_calibration(tracks, qb_track, snap_frame, y_sign):
    """Return (los_pixel_x, yards_per_pixel_x) for sideline Y axis."""
    los_px = None
    if qb_track and qb_track["positions"]:
        nf     = min(qb_track["positions"], key=lambda f: abs(f-snap_frame))
        los_px = qb_track["positions"][nf][0]
    # Fallback: none
    return los_px, None   # calibration deferred until yard-line Hough is added


def get_x_calibration(tracks, ol_ids, snap_approx):
    """Return OL centroid X pixel in end zone view = X=0 reference."""
    ol_positions = []
    for t in tracks.values():
        if t["id"] not in ol_ids or not t["positions"]: continue
        nf = min(t["positions"], key=lambda f: abs(f-snap_approx))
        ol_positions.append(t["positions"][nf][0])
    if not ol_positions: return None
    return float(np.mean(ol_positions))



# COLOR BOUNDARY DETECTION

def find_horizontal_color_split(crop_bgr, min_color=0.10):
    """Scan the jersey zone of a bbox crop for a yellow↔blue color boundary.
    Returns the x-column within the crop where the dominant color switches,
    or None if no clear boundary exists.
    Only fires when both yellow and blue are significantly present (merged box).
    min_color can be lowered (e.g. 0.07) for QB-proximal bboxes where a pass
    rusher may be mostly occluded by an OL."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    j_top = int(h * 0.15)
    j_bot = int(h * 0.70)
    jersey = crop_bgr[j_top:j_bot, :]
    if jersey.shape[0] < 4:
        return None
    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    yel_mask = cv2.inRange(hsv, HSV_YEL_LO, HSV_YEL_HI)
    blu_mask  = cv2.inRange(hsv, HSV_BLU_LO, HSV_BLU_HI)
    jh = jersey.shape[0]
    yel_col = yel_mask.sum(axis=0).astype(float) / (jh * 255)
    blu_col  = blu_mask.sum(axis=0).astype(float) / (jh * 255)
    # Both colors must be meaningfully present before we trust a split
    if yel_col.max() < min_color or blu_col.max() < min_color:
        return None
    smooth = max(3, w // 8)
    kernel = np.ones(smooth) / smooth
    yel_s = np.convolve(yel_col, kernel, mode='same')
    blu_s  = np.convolve(blu_col, kernel, mode='same')
    diff = yel_s - blu_s
    # Sign changes mark transitions between dominant colors
    sign_changes = [i for i in range(1, w) if diff[i - 1] * diff[i] < 0]
    if not sign_changes:
        return None
    # Pick the sharpest transition
    best_x = max(sign_changes, key=lambda x: abs(float(diff[x]) - float(diff[x - 1])))
    # Reject transitions at extreme edges (< 15% or > 85% of width)
    if best_x < int(w * 0.15) or best_x > int(w * 0.85):
        return None
    return best_x


def find_blob_centroids(crop_bgr):
    """
    When a single YOLO box contains both a receiver and a defender, find the
    centroid of the largest yellow blob and the largest blue blob within the
    jersey zone (20-48% height).

    Returns (yel_cx, yel_cy, blu_cx, blu_cy) in crop-local pixel coordinates,
    or None if either colour has no meaningful blob (defender fully occluded).

    Used as fallback when find_horizontal_color_split cannot find a clean
    left-right boundary (e.g. defender directly behind receiver).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    j_top = int(h * 0.20)
    j_bot = int(h * 0.48)
    if j_bot <= j_top or w < 6:
        return None
    jersey = crop_bgr[j_top:j_bot, :]
    hsv_j  = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

    def _blob_centroid(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 30:   # too small - noise
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        return M["m10"] / M["m00"], M["m01"] / M["m00"]

    yel = _blob_centroid(cv2.inRange(hsv_j, HSV_YEL_LO, HSV_YEL_HI))
    blu = _blob_centroid(cv2.inRange(hsv_j, HSV_BLU_LO, HSV_BLU_HI))

    if yel is None or blu is None:
        return None

    # Convert jersey-zone local coords back to crop-local coords
    return yel[0], yel[1] + j_top, blu[0], blu[1] + j_top


# DRAW HELPERS


def draw_box(frame, x1, y1, x2, y2, tid, team, jcounts, shoe_score,
             role=None, coord_label=None, dim=False):
    col = COL.get(team, COL["UNKNOWN"])
    if dim:
        cv2.rectangle(frame, (x1,y1),(x2,y2),(60,60,60),1)
        cv2.putText(frame,"OOB",(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.35,(60,60,60),1)
        return
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
    label = role if role else f"#{tid} {team[:3]}"
    if coord_label:
        label = f"{label}  {coord_label}"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame,(x1,y1-th-4),(x1+tw+2,y1),col,-1)
    cv2.putText(frame,label,(x1+1,y1-3),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),1,cv2.LINE_AA)
    # Jersey colour bar
    bar_y, bx = y1+2, x1
    total_px  = max(sum(jcounts.values()),1)
    for tname,tpx in jcounts.items():
        w = int((tpx/total_px)*(x2-x1))
        if w>0:
            cv2.rectangle(frame,(bx,bar_y),(bx+w,bar_y+4),COL.get(tname,COL["UNKNOWN"]),-1)
            bx += w
    # Shoe/glove bar
    sw = int(shoe_score*(x2-x1))
    if sw>0:
        cv2.rectangle(frame,(x1,bar_y+5),(x1+sw,bar_y+7),(255,255,255),-1)


def draw_banner(frame, text, col, row=0):
    H,W = frame.shape[:2]
    y   = 32 + row*30
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
    cv2.rectangle(frame,(W//2-tw//2-6,y-th-4),(W//2+tw//2+6,y+4),(0,0,0),-1)
    cv2.putText(frame,text,(W//2-tw//2,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2,cv2.LINE_AA)


def draw_oob_line(frame, oob_y, W):
    if oob_y is None: return
    cv2.line(frame,(0,oob_y),(W,oob_y),OOB_LINE_COL,2)
    cv2.putText(frame,"OOB",(8,oob_y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,OOB_LINE_COL,1)


# DEFENDER GHOST POSITIONS

def build_ghost_positions(track, fps, short_gap_s=0.25):
    """
    Fill tracking gaps in a defender track with estimated positions.
      Short gap  (<short_gap_s): carry the last known position forward.
        These gaps are NOT added to display_frames - quick flickers don't
        need ghost circles; the track reconnects on its own.
      Long gap (>=short_gap_s): compute pre-gap velocity and project forward
        with damping, blending toward the post-gap real position.
        Only added to display_frames when the pre-gap context spans >= 0.25s
        so we never project a direction from insufficient data.

    Returns:
      filled        - {frame: (cx, cy)} for every frame first→last seen.
                      Used by the openness calculation.
      display_frames - set of frames that should render a ghost circle.
                      Subset of filled: long gaps with reliable direction only.
    """
    positions = track["positions"]
    if not positions:
        return {}, set()
    max_carry      = int(fps * short_gap_s)       # ~14 frames - short gap threshold
    min_ctx_frames = int(fps * 0.25)              # need 0.25s of history to trust velocity
    vel_window     = max(2, int(fps * 0.25))      # use 0.25s window for velocity
    frames_seen    = sorted(positions.keys())
    filled         = dict(positions)
    display_frames = set()

    for i in range(len(frames_seen) - 1):
        f_last = frames_seen[i]
        f_next = frames_seen[i + 1]
        gap    = f_next - f_last - 1
        if gap == 0:
            continue
        p_last = positions[f_last]
        p_next = positions[f_next]

        if gap <= max_carry:
            # Short gap - carry position, no ghost circle
            for f in range(f_last + 1, f_next):
                filled[f] = p_last
        else:
            # Long gap - project with velocity
            pre = [f for f in frames_seen[:i + 1] if f >= f_last - vel_window]
            has_context = (len(pre) >= 2 and
                           (pre[-1] - pre[0]) >= min_ctx_frames)
            if has_context:
                dt = max(pre[-1] - pre[0], 1)
                vx = (positions[pre[-1]][0] - positions[pre[0]][0]) / dt
                vy = (positions[pre[-1]][1] - positions[pre[0]][1]) / dt
            else:
                vx, vy = 0.0, 0.0   # not enough context - carry without direction
            for j, f in enumerate(range(f_last + 1, f_next), 1):
                alpha  = j / (gap + 1)
                damp   = 1.0 - alpha      # velocity fades out toward p_next
                proj_x = p_last[0] + vx * j * damp
                proj_y = p_last[1] + vy * j * damp
                filled[f] = ((1 - alpha) * proj_x + alpha * p_next[0],
                             (1 - alpha) * proj_y + alpha * p_next[1])
                # Only show circle if we had enough context to trust the direction
                if has_context:
                    display_frames.add(f)

    return filled, display_frames


# FAR-FIELD SUPPLEMENTAL SCAN

def supplemental_far_field_scan(video_path, snap_frame, sl_tracks, model, device,
                                  far_oob_row, oob_y, fps,
                                  conf=0.12, upscale=2, min_offense_expected=9):
    """
    After main tracking, if fewer OFFENSE tracks than expected are visible near
    snap_frame, scan the far-field strip (top 40% of field) at 2x upscale and
    low conf to catch receivers YOLO missed at normal resolution.
    Any detected person not already covered by an existing OFFENSE track within
    80px is injected as a new synthetic OFFENSE track into sl_tracks.
    min_offense_expected = OL(5) + QB(1) + receivers(3-4) ≈ 9.
    """
    if snap_frame is None or far_oob_row is None or oob_y is None:
        return sl_tracks

    # Count OFFENSE tracks visible within 45 frames of snap_frame
    visible = sum(
        1 for t in sl_tracks.values()
        if t["team"] == "OFFENSE"
        and any(abs(f - snap_frame) <= 45 for f in t["positions"])
    )
    if visible >= min_offense_expected:
        print(f"  Far-field scan: {visible} OFFENSE tracks near snap - skipping")
        return sl_tracks
    print(f"  Far-field scan: only {visible} OFFENSE tracks near snap, scanning top field...")

    field_h  = oob_y - far_oob_row
    scan_y1  = max(0, far_oob_row - 10)
    scan_y2  = far_oob_row + int(field_h * 0.42)   # top 42% of field

    # Sample frames bracketing snap_frame
    sample_frames = sorted(set(range(
        max(0, snap_frame - 40), snap_frame + 40, 6)))

    new_dets   = defaultdict(dict)   # pseudo_id -> {frame: (cx, cy)}
    next_pid   = max(sl_tracks.keys(), default=0) + 2000

    cap = cv2.VideoCapture(str(video_path))
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        H_f, W_f = frame.shape[:2]
        y1 = min(scan_y1, H_f - 1)
        y2 = min(scan_y2, H_f)
        strip = frame[y1:y2, :]
        if strip.shape[0] < 10:
            continue

        strip_up = cv2.resize(strip,
                               (strip.shape[1] * upscale, strip.shape[0] * upscale),
                               interpolation=cv2.INTER_CUBIC)

        res = model.predict(strip_up, classes=[0], conf=conf, iou=0.30,
                             device=device, verbose=False)
        if not res or res[0].boxes is None:
            continue

        for (bx1, by1, bx2, by2) in res[0].boxes.xyxy.cpu().numpy():
            cx = (bx1 + bx2) / 2 / upscale
            cy = y1 + (by1 + by2) / 2 / upscale

            # Skip if already covered by an existing OFFENSE track
            covered = False
            for t in sl_tracks.values():
                if t["team"] != "OFFENSE":
                    continue
                nf = min((f for f in t["positions"]), key=lambda f: abs(f - fi),
                         default=None)
                if nf is not None and abs(nf - fi) <= 6:
                    d = math.hypot(cx - t["positions"][nf][0],
                                   cy - t["positions"][nf][1])
                    if d < 80:
                        covered = True
                        break
            if covered:
                continue

            # Assign to nearest existing synthetic track or open a new one
            best_pid, best_d = None, float("inf")
            for pid, ppos in new_dets.items():
                if not ppos:
                    continue
                nf2 = min(ppos, key=lambda f: abs(f - fi))
                if abs(nf2 - fi) <= 12:
                    d = math.hypot(cx - ppos[nf2][0], cy - ppos[nf2][1])
                    if d < 150 and d < best_d:
                        best_d, best_pid = d, pid
            if best_pid is not None:
                new_dets[best_pid][fi] = (cx, cy)
            else:
                new_dets[next_pid][fi] = (cx, cy)
                next_pid += 1
    cap.release()

    added = 0
    for pid, positions in new_dets.items():
        if len(positions) < 3:          # need at least 3 frames
            continue
        sl_tracks[pid] = {
            "id":         pid,
            "team":       "OFFENSE",
            "positions":  dict(positions),
            "bboxes":     {},
            "shoe_score": 0.0,
            "qb_signals": 0,
            "synthetic":  True,   # injected by supplemental scan - no sl_dets entry
        }
        added += 1
        print(f"    Synthetic OFFENSE track #{pid}: {len(positions)} frames near snap")

    print(f"  Far-field scan done: {added} synthetic track(s) added")
    return sl_tracks



# MAIN

# CSV OUTPUT


def detect_merged_defender_positions(video_path, sl_dets, sl_tracks,
                                     track_merge_map, seg_start, seg_end,
                                     snap_frame=None):
    """
    Sequential frame pass: for every OFFENSE detection where find_horizontal_color_split
    fires, classify each half to find the DEFENSE side and record its centroid as a
    synthetic DEFENSE track position.

    If split fires but neither half classifies as DEFENSE, the full-box centroid is
    used - placing the synthetic defender at the receiver's own centroid, which makes
    dist_between(...) return ≈ 0 (the "zero fallback").

    Post-snap, OFFENSE bboxes within QB_PROX_PX pixels of the QB get a lower
    color-split threshold (0.07 vs 0.10) so partially-occluded pass rushers
    engaged with OL are detected more aggressively.

    Returns {synth_id: track_dict} ready to be merged into sl_tracks.
    Synth IDs are 90000 + eff_track_id to avoid collision with YOLO IDs.
    """
    QB_PROX_PX   = 200   # horizontal pixel radius around QB for aggressive detection
    MIN_COLOR_STD = 0.10  # default split threshold
    MIN_COLOR_QB  = 0.07  # looser threshold near QB post-snap

    # Build per-frame QB cx map for proximity check
    qb_cx_map = {}
    for t in sl_tracks.values():
        if t["team"] != "QB": continue
        for f, pos in t["positions"].items():
            qb_cx_map[f] = pos[0]

    synth = {}
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
    fi = seg_start
    while fi <= seg_end:
        ret, frame = cap.read()
        if not ret: break
        qb_cx_fi = qb_cx_map.get(fi)
        for (tid, x1, y1, x2, y2, _t, jc, shoe, is_oob, _qs) in sl_dets.get(fi, []):
            if is_oob or y2 <= y1 or x2 <= x1: continue
            eff = track_merge_map.get(tid, tid)
            if eff not in sl_tracks or sl_tracks[eff]["team"] != "OFFENSE": continue
            crop = frame[y1:y2, x1:x2]
            # Stage 1: left-right colour split
            # Near the QB post-snap, lower the threshold so partially-occluded
            # pass rushers engaged with an OL are detected.
            bbox_cx = (x1 + x2) / 2.0
            near_qb = (snap_frame is not None and fi > snap_frame and
                       qb_cx_fi is not None and abs(bbox_cx - qb_cx_fi) < QB_PROX_PX)
            mc = MIN_COLOR_QB if near_qb else MIN_COLOR_STD
            split_x  = find_horizontal_color_split(crop, min_color=mc)
            def_cx = def_cy = None
            if split_x is not None:
                abs_split = x1 + split_x
                lt, _, _, _ = classify_player(frame[y1:y2, x1:abs_split])
                rt, _, _, _ = classify_player(frame[y1:y2, abs_split:x2])
                if lt == "DEFENSE" and rt == "OFFENSE":
                    def_cx = (x1 + abs_split) / 2.0
                    def_cy = (y1 + y2) / 2.0
                elif rt == "DEFENSE" and lt == "OFFENSE":
                    def_cx = (abs_split + x2) / 2.0
                    def_cy = (y1 + y2) / 2.0

            # Stage 2: blob centroid (defender behind receiver)
            if def_cx is None:
                blobs = find_blob_centroids(crop)
                if blobs is not None:
                    _, _, blu_lx, blu_ly = blobs
                    def_cx = x1 + blu_lx
                    def_cy = y1 + blu_ly

            # Stage 3: nearest confirmed DEFENSE track within 150px
            # Hard distance cap: only use a nearby defender, never grab one
            # from across the field - that scatters synthetic positions and
            # causes ghost paths to fly off into nowhere.
            if def_cx is None:
                recv_cx = (x1 + x2) / 2.0
                recv_cy = (y1 + y2) / 2.0
                best_d, best_pos = float("inf"), None
                for dt in sl_tracks.values():
                    if dt["team"] != "DEFENSE": continue
                    dpos = dt["positions"].get(fi)
                    if dpos is None: continue
                    d = math.hypot(recv_cx - dpos[0], recv_cy - dpos[1])
                    if d < best_d:
                        best_d, best_pos = d, dpos
                if best_pos is not None and best_d < 150:
                    def_cx, def_cy = best_pos
                else:
                    continue   # no nearby defender - skip this frame
            synth_id = 90000 + eff
            if synth_id not in synth:
                synth[synth_id] = {
                    "id": synth_id, "team": "DEFENSE",
                    "positions": {}, "bboxes": {},
                    "shoe_score": 0.0, "synthetic": True,
                }
            synth[synth_id]["positions"][fi] = (def_cx, def_cy)
        fi += 1
    cap.release()
    return synth


def write_annotate_csvs(clip_day, play_number, fps,
                        snap_frame, throw_frame,
                        sl_tracks, qb_label_map, recv_id_to_label,
                        ol_ids, los_px, y_sign, seg_end, out_dir,
                        lateral_cal=None, px_per_yd_depth=None,
                        defender_ghosts=None):
    """
    Write two CSVs mirroring CV Test Results / CV Test Time Series.
    Openness is computed from 1s post-snap onward (before that reflects formation
    alignment, not route separation). Defender positions are ghost-filled across
    tracking gaps so openness doesn't spike when a defender drops out briefly.
    """
    if snap_frame is None:
        print("  No snap - skipping CSV output"); return

    defense   = [t for t in sl_tracks.values() if t["team"] == "DEFENSE"]
    qb_ids    = set(qb_label_map.keys())
    open_start = snap_frame + int(fps * 1.0)   # openness valid from 1s post-snap
    end_frame = throw_frame if throw_frame else seg_end

    label_tids = defaultdict(set)
    for tid, lbl in recv_id_to_label.items():
        label_tids[lbl].add(tid)
    recv_labels = sorted(label_tids)

    calibrated = lateral_cal is not None and px_per_yd_depth is not None
    unit       = "yds" if calibrated else "px"

    # coordinate helpers
    def to_coords(pos):
        """(cx, cy) pixel → (y_depth_yds, x_lateral_yds) in yards or pixels.
        cx = horizontal pixel = downfield (field Y) - linear scale DEPTH_PX_PER_YD
        cy = vertical pixel   = lateral  (field X) - perspective-corrected lateral_cal
        """
        cx, cy = pos
        if calibrated:
            y = round((cx - los_px) * y_sign / DEPTH_PX_PER_YD, 2) if los_px else 0.0
            x = round(lateral_cal(cy), 2)
        else:
            y = round((cx - los_px) * y_sign, 1) if los_px else round(cx, 1)
            x = round(cy, 1)
        return y, x

    def dist_between(p1, p2):
        """Euclidean distance in yards (or pixels if uncalibrated)."""
        if calibrated:
            y1, x1 = to_coords(p1)
            y2, x2 = to_coords(p2)
            return math.hypot(y1 - y2, x1 - x2)
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def press_dist(p1, p2):
        """Downfield-only distance for pressure (yards or pixels).
        The sideline cam measures the horizontal (downfield) axis cleanly via
        DEPTH_PX_PER_YD. The lateral axis uses perspective calibration whose
        small pixel errors map to large yard errors (~4-5 px/yd vs 45 px/yd
        for depth), inflating 2-D Euclidean pressure by 2-3x. Pressure is
        physically about how many downfield yards a rusher must still close,
        so depth-only is both more accurate and more meaningful here.
        """
        if calibrated and los_px:
            return abs(p1[0] - p2[0]) / DEPTH_PX_PER_YD
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def nearest_def(pos, fi, allowed_synth=None):
        """Ghost-filled nearest defender. Returns (distance, defender_id) or (None, None).
        Synthetic tracks (id >= 90000) are only considered if their id is in allowed_synth
        - prevents a synth created for one receiver from matching a different receiver."""
        if fi < open_start:
            return None, None
        best_d, best_id = None, None
        for dt in defense:
            if dt["id"] >= 90000 and (allowed_synth is None or dt["id"] not in allowed_synth):
                continue
            ghost = defender_ghosts.get(dt["id"]) if defender_ghosts else None
            dp    = ghost.get(fi) if ghost else dt["positions"].get(fi)
            if dp is None: continue
            d = dist_between(pos, dp)
            if best_d is None or d < best_d:
                best_d, best_id = d, dt["id"]
        return best_d, best_id

    def recv_pos(lbl, fi):
        for tid in label_tids[lbl]:
            p = sl_tracks.get(tid, {}).get("positions", {}).get(fi)
            if p: return p
        return None

    # TIME SERIES - Pass 1: build raw openness per receiver
    # Raw values are unstable because the "nearest defender" can flip frame to
    # frame when a defender briefly loses tracking.  We smooth them in Pass 2.
    raw_open = {lbl: {} for lbl in recv_labels}   # {label: {fi: openness}}
    raw_def  = {lbl: {} for lbl in recv_labels}   # {label: {fi: defender_id}}
    raw_pos  = {lbl: {} for lbl in recv_labels}   # {label: {fi: (y_val, x_val)}}
    # Synthetic tracks (90000+eff) are receiver-specific - only allow them for the
    # receiver whose eff they were created from, not for every other receiver.
    label_allowed_synth = {lbl: {90000 + tid for tid in tids}
                           for lbl, tids in label_tids.items()}
    # Hysteresis: once a synthetic (merged-coverage) reading is established,
    # a real defender must beat the last synthetic distance by this margin to
    # override it.  Prevents noisy flips to a distant real defender when the
    # color-split fires only on some frames.
    SYNTH_HYSTERESIS_YD  = 1.5
    SYNTH_HYSTERESIS_EXP = int(fps * 0.5)   # forget after 0.5 s of no synthetic
    last_synth_d  = {lbl: None for lbl in recv_labels}
    last_synth_fi = {lbl: -9999 for lbl in recv_labels}
    for fi in range(snap_frame, end_frame + 1):
        for lbl in recv_labels:
            pos = recv_pos(lbl, fi)
            if pos is None: continue
            raw_pos[lbl][fi] = to_coords(pos)
            # Expire hysteresis if too long since last synthetic hit
            if (last_synth_d[lbl] is not None and
                    fi - last_synth_fi[lbl] > SYNTH_HYSTERESIS_EXP):
                last_synth_d[lbl] = None
            d, def_id = nearest_def(pos, fi, label_allowed_synth.get(lbl, set()))
            if d is None:
                # No separate defender track found - receiver bbox must contain
                # the defender (merged).  Treat as 0.0 yd (tight coverage).
                if fi >= open_start:
                    raw_open[lbl][fi] = 0.0
                continue
            # Hysteresis: real defender must clear last synthetic distance + margin
            if (def_id is not None and def_id < 90000 and
                    last_synth_d[lbl] is not None and
                    d > last_synth_d[lbl] + SYNTH_HYSTERESIS_YD):
                # Distant real defender while merged coverage was active -
                # carry the last synthetic distance rather than leaving blank.
                raw_open[lbl][fi] = last_synth_d[lbl]
                continue
            raw_open[lbl][fi] = d
            raw_def[lbl][fi]  = def_id
            if def_id is not None and def_id >= 90000:
                last_synth_d[lbl]  = d
                last_synth_fi[lbl] = fi

    # Pass 2: rolling-median smoothing over ~0.15s window
    # 0.15s catches single-frame anomalies (defender flip) while still tracking
    # genuine route-running changes which happen over multiple frames.
    SMOOTH_WINDOW = max(3, int(fps * 0.15))
    half_w = SMOOTH_WINDOW // 2
    smoothed_open = {lbl: {} for lbl in recv_labels}
    for lbl in recv_labels:
        sorted_frames = sorted(raw_open[lbl])
        for i, fi in enumerate(sorted_frames):
            lo = max(0, i - half_w)
            hi = min(len(sorted_frames), i + half_w + 1)
            window = [raw_open[lbl][sorted_frames[j]] for j in range(lo, hi)]
            window.sort()
            smoothed_open[lbl][fi] = window[len(window) // 2]   # median

    # Pass 3: emit rows
    ts_rows = []
    for fi in range(snap_frame, end_frame + 1):
        t_s = round((fi - snap_frame) / fps, 2)
        for lbl in recv_labels:
            if fi not in raw_pos[lbl]: continue
            y_val, x_val = raw_pos[lbl][fi]
            openness = smoothed_open[lbl].get(fi)
            ts_rows.append({
                "Day":               clip_day,
                "Play_Number":       play_number,
                "Time_Post_Snap_s":  t_s,
                "Receiver":          lbl,
                f"Openness_{unit}":  round(openness, 2) if openness is not None else "",
                "Defender_ID":       raw_def[lbl].get(fi, ""),
                f"Y_{unit}":         y_val,
                f"X_{unit}":         x_val,
            })

    # RESULTS
    PRESSURE_THR = 2.0 if calibrated else 150  # 2 yards or ~150 px

    peak_open, peak_t, open_at_throw = {}, {}, {}
    for lbl in recv_labels:
        best_d, best_t, at_throw = None, None, None
        # Use smoothed values for results so peak/throw openness reflects the
        # stabilised signal, not single-frame anomalies.
        for fi in sorted(smoothed_open[lbl]):
            d   = smoothed_open[lbl][fi]
            t_s = (fi - snap_frame) / fps
            if best_d is None or d > best_d:
                best_d, best_t = d, t_s
            if throw_frame and fi == throw_frame:
                at_throw = d
        peak_open[lbl]     = round(best_d, 2) if best_d   is not None else ""
        peak_t[lbl]        = round(best_t, 2) if best_t   is not None else ""
        open_at_throw[lbl] = round(at_throw, 2) if at_throw is not None else ""

    # PRESSURE TIME SERIES
    # QB to nearest DEFENSE track (real + synthetic OL/DL engagement merges).
    # Uses pressure-only static-carry ghost-fill: if a defender lost detection
    # for ≤5 frames, carry last known position forward. 5-frame cap (~0.09s)
    # limits blitzing-DB position error to ~0.4yds while covering normal
    # single-frame YOLO dropouts. Does NOT touch defender_ghosts or openness data.
    press_start   = snap_frame + int(fps * 0.5)
    def _press_def_pos(dt, fi):
        """Raw position only - no carry. Log whoever is detected in this frame."""
        return dt["positions"].get(fi)

    # OL tracks for proxy pressure
    ol_tracks = [t for t in sl_tracks.values() if t["id"] in ol_ids]

    # Pass A: raw per-frame distances - real defenders + nearest OL proxy
    # Synthetic tracks (id ≥ 90000) excluded from defender pass: OL blue pants
    # trigger blob detection and place a phantom ~1-2yd from QB.
    # OL proxy: if an OL is being pushed close to the QB, a pass rusher is
    # right behind them. Same static carry, same gap limit.
    raw_press    = {}
    raw_press_id = {}
    raw_press_n  = {}
    raw_press_qb = {}
    raw_ol_press = {}
    raw_ol_id    = {}
    for fi in range(press_start, end_frame + 1):
        qb_pos = None
        for qid in qb_ids:
            qb_pos = sl_tracks.get(qid, {}).get("positions", {}).get(fi)
            if qb_pos: break
        if not qb_pos: continue
        # Nearest real defender
        best_d, best_id = None, None
        n_near = 0
        for dt in defense:
            if dt["id"] >= 90000:
                continue
            dp = _press_def_pos(dt, fi)
            if dp is None: continue
            d = press_dist(qb_pos, dp)
            if best_d is None or d < best_d:
                best_d, best_id = d, dt["id"]
            if d < PRESSURE_THR:
                n_near += 1
        # Nearest OL proxy
        ol_d, ol_id = None, None
        for olt in ol_tracks:
            dp = _press_def_pos(olt, fi)
            if dp is None: continue
            d = press_dist(qb_pos, dp)
            if ol_d is None or d < ol_d:
                ol_d, ol_id = d, olt["id"]
        if best_d is None and ol_d is None: continue
        raw_press_qb[fi] = qb_pos
        if best_d is not None:
            raw_press[fi]    = best_d
            raw_press_id[fi] = best_id
            raw_press_n[fi]  = n_near
        if ol_d is not None:
            raw_ol_press[fi] = ol_d
            raw_ol_id[fi]    = ol_id

    # Pass B: 0.15s rolling-median smoothing on both signals
    PRESS_SMOOTH = max(3, int(fps * 0.15))
    half_pw      = PRESS_SMOOTH // 2

    def _smooth(raw_dict):
        keys = sorted(raw_dict)
        out  = {}
        for i, fi in enumerate(keys):
            lo = max(0, i - half_pw)
            hi = min(len(keys), i + half_pw + 1)
            w  = sorted([raw_dict[keys[j]] for j in range(lo, hi)])
            out[fi] = w[len(w) // 2]
        return out

    smoothed_press = _smooth(raw_press)
    smoothed_ol    = _smooth(raw_ol_press)

    # Pass C: emit rows + build summary - effective pressure = min of both signals
    press_rows  = []
    max_press_dist, max_press_t, max_press_n = None, None, 0
    all_pf = sorted(set(smoothed_press) | set(smoothed_ol))
    for fi in all_pf:
        sd_def = smoothed_press.get(fi)
        sd_ol  = smoothed_ol.get(fi)
        if sd_def is None and sd_ol is None: continue
        sd_eff = min(v for v in [sd_def, sd_ol] if v is not None)
        qb_y, qb_x = to_coords(raw_press_qb[fi])
        press_rows.append({
            "Day":                          clip_day,
            "Play_Number":                  play_number,
            "Time_Post_Snap_s":             round((fi - snap_frame) / fps, 2),
            f"Effective_Pressure_{unit}":   round(sd_eff, 2),
            f"Defender_Pressure_{unit}":    round(sd_def, 2) if sd_def is not None else "",
            "Nearest_Defender_ID":          raw_press_id.get(fi, ""),
            f"OL_Pressure_{unit}":          round(sd_ol, 2) if sd_ol is not None else "",
            "Nearest_OL_ID":                raw_ol_id.get(fi, ""),
            f"QB_Y_{unit}":                 qb_y,
            f"QB_X_{unit}":                 qb_x,
        })
        if max_press_dist is None or sd_eff < max_press_dist:
            max_press_dist = sd_eff
            max_press_t    = round((fi - snap_frame) / fps, 2)
            max_press_n    = raw_press_n.get(fi, 0)

    res_row = {
        "Day":                              clip_day,
        "Play_Number":                      play_number,
        "Snap_Frame":                       snap_frame,
        "Throw_Frame":                      throw_frame if throw_frame else "",
        "Time_To_Throw_s":                  round((throw_frame-snap_frame)/fps, 2) if throw_frame else "",
        "Calibrated":                       "Y" if calibrated else "N",
        "QB_Tracked":                       "Y" if qb_ids else "N",
        "OL_Count":                         len(ol_ids),
        "Receivers_Detected":               len(recv_labels),
        # Pressure metrics temporarily disabled - re-enable when detection is stable
        # "Max_Pressure_Defenders_Near":              max_press_n,
        # f"Max_Effective_Pressure_{unit}":           round(max_press_dist, 2) if max_press_dist else "",
        # "Max_Effective_Pressure_Time_Post_Snap_s":  max_press_t if max_press_t else "",
    }
    for lbl in ["R1", "R2", "R3", "R4", "R5"]:
        res_row[f"{lbl}_Openness_At_Throw_{unit}"] = open_at_throw.get(lbl, "")
        res_row[f"{lbl}_Peak_Openness_{unit}"]      = peak_open.get(lbl, "")
        res_row[f"{lbl}_Peak_Time_Post_Snap_s"]     = peak_t.get(lbl, "")
        res_row[f"{lbl}_Tracking_Lost"]             = "N" if lbl in recv_labels else "Y"

    # WRITE
    ts_path    = out_dir / "Day3_annotate_timeseries.csv"
    res_path   = out_dir / "Day3_annotate_results.csv"
    press_path = out_dir / "Day3_annotate_pressure_timeseries.csv"

    if ts_rows:
        with open(ts_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()))
            w.writeheader(); w.writerows(ts_rows)
        print(f"  Time series: {ts_path}  ({len(ts_rows)} rows)")

    # Pressure timeseries disabled - re-enable when detection is stable
    # if press_rows:
    #     with open(press_path, "w", newline="") as f:
    #         w = csv.DictWriter(f, fieldnames=list(press_rows[0].keys()))
    #         w.writeheader(); w.writerows(press_rows)
    #     print(f"  Pressure TS: {press_path}  ({len(press_rows)} rows)")

    with open(res_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res_row.keys()))
        w.writeheader(); w.writerow(res_row)
    print(f"  Results:     {res_path}")




def main():
    if not HAS_YOLO:
        print("ultralytics not installed."); return

    cap0 = cv2.VideoCapture(str(CLIP))
    fps   = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()
    print(f"Clip: {total} frames @ {fps:.1f} fps  ({W}×{H})")

    model  = YOLO(YOLO_MODEL)
    device = get_device()
    print(f"Device: {device}")

    # 1. HARD CUTS
    print("Detecting hard cuts...")
    cuts = detect_hard_cuts(CLIP, total)
    print(f"  Cuts: {cuts}")
    # Segment order confirmed for this clip family:
    #   0 = elevated sideline
    #   1 = EZ zoomed-in   (SKIP)
    #   2 = EZ zoomed-out  (annotate)
    seg_sl     = (0,           cuts[0]-1)
    seg_ez_in  = (cuts[0],     cuts[1]-1)
    seg_ez_out = (cuts[1],     total-1)
    print(f"  SL {seg_sl}  EZ-in {seg_ez_in}  EZ-out {seg_ez_out}")

    # 2. OOB LINE + LANDMARK CALIBRATION
    print("Detecting OOB line (green field edge)...")
    oob_y = detect_oob_line_green(CLIP, seg_sl[0], H, W)
    print(f"  OOB Y: {oob_y}")

    print("Detecting field landmarks for yard calibration...")
    far_oob_row, near_hash_row, far_hash_row = detect_field_landmarks(
        CLIP, seg_sl[0], oob_y, W) if oob_y else (None, None, None)
    print(f"  Far OOB: {far_oob_row}  Near hash: {near_hash_row}  Far hash: {far_hash_row}")

    lateral_cal    = build_lateral_cal(oob_y, far_oob_row, near_hash_row, far_hash_row) \
                     if far_oob_row else None
    px_per_yd_depth = (far_oob_row and oob_y) and (oob_y - far_oob_row) / 65.0 or None
    if lateral_cal:
        print(f"  Lateral calibration active  ({65/(oob_y-far_oob_row)*100:.1f} px/yd approx)")
    else:
        print("  Lateral calibration unavailable - falling back to pixels")

    # 3. TRACK SIDELINE
    print(f"Tracking sideline [{seg_sl[0]}→{seg_sl[1]}]...")
    sl_tracks, sl_dets = track_segment(CLIP, seg_sl[0], seg_sl[1],
                                        model, device, oob_y=oob_y,
                                        far_oob_row=far_oob_row,
                                        seed_frames=int(fps * 1.5))
    print(f"  {len(sl_tracks)} raw tracks")
    run_jersey_ocr(CLIP, sl_tracks, seg_sl[0], seg_sl[1])

    # 4. SNAP (on raw tracks - consolidation gaps would break stillness calc)
    print("Rough OL cluster for snap detection...")
    yellow_ids_raw = [t["id"] for t in sl_tracks.values() if t["team"] == "OFFENSE"]
    print("Detecting snap...")
    snap_frame = detect_snap_ol_stillness(sl_tracks, set(yellow_ids_raw),
                                           seg_sl[0], seg_sl[1], fps)
    print(f"  Snap frame (raw): {snap_frame}")
    if snap_frame is not None:
        snap_frame = max(seg_sl[0], snap_frame - SNAP_LEAD_FRAMES)
    print(f"  Snap frame (adjusted −{SNAP_LEAD_FRAMES}f): {snap_frame}")

    # 4b. CONSOLIDATE (after snap so stillness calc isn't corrupted by gaps)
    # Merge fragmented OFFENSE tracks so OL/TE/receiver movement checks see the
    # full positional history across the entire clip, not just short fragments.
    print("Consolidating fragmented OFFENSE tracks...")
    sl_tracks, track_merge_map = consolidate_offense_tracks(
        sl_tracks, seg_sl[1], fps)
    n_off = sum(1 for t in sl_tracks.values() if t["team"] == "OFFENSE")
    print(f"  Merged {len(track_merge_map)} fragments → {n_off} OFFENSE tracks remain")

    # Supplemental far-field scan - catches receivers YOLO missed at normal res
    sl_tracks = supplemental_far_field_scan(
        CLIP, snap_frame, sl_tracks, model, device,
        far_oob_row=far_oob_row, oob_y=oob_y, fps=fps)

    # 5. OL BY POST-SNAP STILLNESS
    print("Identifying OL by post-snap stillness...")
    ol_ids = identify_ol_post_snap(sl_tracks, snap_frame, fps)
    print(f"  OL IDs: {ol_ids}")

    # 5b. RECEIVER COUNT
    n_recv = FORMATION_RECV_COUNT

    # 6. QB + RECEIVERS (retroactive blocker eviction)
    qb_sl = find_qb_sideline(sl_tracks, snap_frame, ol_ids)
    print(f"  QB sideline: {qb_sl['id'] if qb_sl else 'NOT FOUND'}")
    receiver_map = number_receivers(sl_tracks, qb_sl, snap_frame,
                                     ol_ids, n_recv, fps)
    print(f"  Initial receivers: {receiver_map}")

    # Retroactively evict non-moving receivers and re-run with corrected OL set
    receiver_map, evicted_ids = evict_blocking_receivers(
        receiver_map, sl_tracks, snap_frame, fps)
    if evicted_ids:
        ol_ids = ol_ids | evicted_ids
        print(f"  Re-running with updated OL set: {ol_ids}")
        qb_sl        = find_qb_sideline(sl_tracks, snap_frame, ol_ids)
        receiver_map = number_receivers(sl_tracks, qb_sl, snap_frame,
                                        ol_ids, n_recv, fps)
        # One more eviction pass in case new receivers are also blockers
        receiver_map, extra = evict_blocking_receivers(
            receiver_map, sl_tracks, snap_frame, fps)
        if extra:
            ol_ids    = ol_ids | extra
            qb_sl     = find_qb_sideline(sl_tracks, snap_frame, ol_ids)
            receiver_map = number_receivers(sl_tracks, qb_sl, snap_frame,
                                            ol_ids, n_recv, fps)
    print(f"  Receivers: {receiver_map}")

    recv_id_to_label = {v:k for k,v in receiver_map.items()}
    qb_id = qb_sl["id"] if qb_sl else None
    gap   = int(fps * 0.70)   # ~40 frames - wide enough to survive coverage occlusion

    # Stitch receivers: OFFENSE-only (allow DEFENSE caused R1 to flip last time)
    recv_id_to_label = stitch_tracks(
        sl_tracks, recv_id_to_label,
        protected_ids=(ol_ids | {qb_id}) if qb_id else ol_ids,
        end_frame=seg_sl[1], max_gap_frames=gap, max_dist_px=220,
        search_teams=("OFFENSE",))
    print(f"  Stitched receiver map: {recv_id_to_label}")

    # Stitch QB: only one QB in sideline view, so any re-ID is the same player
    qb_label_map = {}
    if qb_sl:
        qb_label_map = stitch_tracks(
            sl_tracks, {qb_id: "QB"},
            protected_ids=ol_ids | set(recv_id_to_label.keys()),
            end_frame=seg_sl[1], max_gap_frames=gap, max_dist_px=150,
            search_teams=("OFFENSE",))
        print(f"  Stitched QB map: {qb_label_map}")

    # 7. Y DIRECTION + CALIBRATION
    y_sign = detect_y_direction(sl_tracks, snap_frame, fps) if snap_frame else 1
    los_px, _ = get_y_calibration(sl_tracks, qb_sl, snap_frame, y_sign)
    print(f"  Y direction sign: {y_sign}   LOS pixel X: {los_px}")

    # 8. THROW
    print("Detecting throw...")
    throw_arm  = detect_throw_arm_motion(CLIP, qb_sl, snap_frame, seg_sl[1], fps,
                                          y_sign=y_sign) if snap_frame else None
    print("  Running post-snap ball detection...")
    ball_dets  = detect_ball_post_snap(CLIP, snap_frame, seg_sl[1], model, device) \
                 if snap_frame else {}
    throw_ball = detect_throw_ball_departure(ball_dets, qb_sl, snap_frame, fps) \
                 if snap_frame else None
    throw_ol   = detect_throw_linemen_drop(sl_tracks, ol_ids, snap_frame,
                                            seg_sl[1], fps) if snap_frame else None
    print(f"  Arm:{throw_arm}  Ball:{throw_ball}  Linemen:{throw_ol}")
    throw_frame = throw_arm or throw_ball or throw_ol
    if throw_frame is not None:
        throw_frame = min(throw_frame + int(fps * 0.25), seg_sl[1])
    print(f"  → Throw: {throw_frame}")

    # 9. EZ ZOOMED-OUT - skipped (sideline-only mode)
    ez_tracks, ez_dets, ez_roles, coaches = {}, {}, {}, set()
    x_ref_px = None

    # 10. ROLE MAPS + TEAM FREEZE
    sl_roles = {}
    sl_roles.update(qb_label_map)
    sl_roles.update(recv_id_to_label)
    for tid in ol_ids:
        if tid not in sl_roles:
            sl_roles[tid] = "OL"

    # Freeze receiver tracks to OFFENSE so color votes from overlapping defenders
    # don't flip the box color or knock a receiver out of stitch eligibility
    for tid in recv_id_to_label:
        if tid in sl_tracks:
            sl_tracks[tid]["team"] = "OFFENSE"

    # 10b. INJECT SPLIT-DETECTED DEFENDERS
    # Scan for OFFENSE bboxes that contain a yellow↔blue split (defender pressed
    # tight enough that YOLO merged the two into one box).  Injects each split
    # defender as a synthetic DEFENSE track so nearest_def() in the CSV and the
    # live openness label both see a real measured distance instead of falling
    # through to the next-nearest defender.  Must run BEFORE ghost-fill so the
    # synthetic positions are interpolated across tracking gaps.
    print("Scanning for merged-box defenders...")
    merged_defs = detect_merged_defender_positions(
        CLIP, sl_dets, sl_tracks, track_merge_map, seg_sl[0], seg_sl[1],
        snap_frame=snap_frame)
    if merged_defs:
        n_frames = sum(len(t["positions"]) for t in merged_defs.values())
        print(f"  Injected {len(merged_defs)} synthetic DEF track(s) "
              f"across {n_frames} merged frames")
        sl_tracks.update(merged_defs)
    else:
        print("  No merged-box defenders detected")

    # 10c-pre. SYNTHETIC TRACK RECEIVER-TRACKING FILL
    # Synthetic defenders (id 90000+eff) live inside the receiver's bbox, so
    # they move with the receiver between frames where color-split fired.
    # Static carry would diverge as the receiver moves; instead project the
    # synthetic by applying the last-known receiver->defender offset onto the
    # receiver's position at each gap frame.  Remaining gaps handled below.
    for synth_id, synth_t in (merged_defs or {}).items():
        source_eff  = synth_id - 90000
        recv_track  = sl_tracks.get(source_eff)
        if recv_track is None:
            continue
        recv_pos_map = recv_track["positions"]
        synth_pos    = synth_t["positions"]          # already in sl_tracks via update
        all_sf = sorted(synth_pos.keys())
        for i in range(len(all_sf) - 1):
            f_last = all_sf[i]
            f_next = all_sf[i + 1]
            if f_next - f_last <= 1:
                continue
            last_sp = synth_pos[f_last]
            last_rp = recv_pos_map.get(f_last)
            if last_rp is None:
                continue
            delta = (last_sp[0] - last_rp[0], last_sp[1] - last_rp[1])
            for f in range(f_last + 1, f_next):
                rp = recv_pos_map.get(f)
                if rp is None:
                    continue
                synth_pos[f] = (rp[0] + delta[0], rp[1] + delta[1])

    # 10c. DEFENDER GHOST POSITIONS
    defender_ghosts      = {}   # {id: {frame: (cx,cy)}}  - full positions for openness
    ghost_display_frames = {}   # {id: set(frames)}        - frames to show a ghost circle
    for t in sl_tracks.values():
        if t["team"] != "DEFENSE": continue
        filled, disp = build_ghost_positions(t, fps)
        defender_ghosts[t["id"]]      = filled
        ghost_display_frames[t["id"]] = disp
    print(f"  Ghost-filled {len(defender_ghosts)} defender tracks")

    # 11. CSV OUTPUT
    print("Writing CSVs...")
    write_annotate_csvs(
        clip_day         = "DAY 3 (03-22-2026)",
        play_number      = 2,
        fps              = fps,
        snap_frame       = snap_frame,
        throw_frame      = throw_frame,
        sl_tracks        = sl_tracks,
        qb_label_map     = qb_label_map,
        recv_id_to_label = recv_id_to_label,
        ol_ids           = ol_ids,
        los_px           = los_px,
        y_sign           = y_sign,
        seg_end          = seg_sl[1],
        out_dir          = CV_DIR,
        lateral_cal      = lateral_cal,
        px_per_yd_depth  = px_per_yd_depth,
        defender_ghosts  = defender_ghosts,
    )

    # 12. RENDER
    print("Rendering...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, fps, (W,H))
    cap_r  = cv2.VideoCapture(str(CLIP))
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fi in range(total):
        ret, frame = cap_r.read()
        if not ret: break

        in_sl     = seg_sl[0]     <= fi <= seg_sl[1]
        in_ez_in  = seg_ez_in[0]  <= fi <= seg_ez_in[1]
        in_ez_out = seg_ez_out[0] <= fi <= seg_ez_out[1]

        if in_ez_in:
            draw_banner(frame, "EZ ZOOMED-IN (skipped)", (80,80,80), row=0)

        elif in_sl:
            draw_oob_line(frame, oob_y, W)
            for (tid,x1,y1,x2,y2,_t,jc,shoe,is_oob,_qs) in sl_dets.get(fi,[]):
                eff        = track_merge_map.get(tid, tid)   # resolve merged track
                final_team = sl_tracks[eff]["team"] if eff in sl_tracks else _t
                role       = sl_roles.get(eff)
                if role == "QB":                     final_team = "QB"
                elif role and role.startswith("R"):  final_team = "OFFENSE"
                # Y coordinate label for receivers / QB
                coord_label = None
                if role and role.startswith("R") and eff in sl_tracks:
                    if fi in sl_tracks[eff]["positions"]:
                        cx, cy = sl_tracks[eff]["positions"][fi]
                        if lateral_cal and los_px:
                            y_yds = round((cx - los_px) * y_sign / DEPTH_PX_PER_YD, 1)
                            x_yds = round(lateral_cal(cy), 1)
                            coord_label = f"Y{y_yds:+.1f}yd X{x_yds:.1f}yd"
                        elif los_px:
                            dy = (cx - los_px) * y_sign
                            coord_label = f"Y≈{dy:.0f}px"
                        # Live openness label - only from 1s post-snap onward
                        if snap_frame and fi >= snap_frame + int(fps):
                            best_open = None
                            my_lbl = recv_id_to_label.get(eff)
                            for def_t in sl_tracks.values():
                                if def_t["team"] != "DEFENSE": continue
                                if def_t["id"] >= 90000:
                                    # Only allow the synthetic tied to this receiver
                                    src_lbl = recv_id_to_label.get(def_t["id"] - 90000)
                                    if src_lbl != my_lbl:
                                        continue
                                ghost = defender_ghosts.get(def_t["id"])
                                dp    = ghost.get(fi) if ghost else \
                                        def_t["positions"].get(fi)
                                if dp is None: continue
                                if lateral_cal:
                                    d_lat = abs(lateral_cal(cy) - lateral_cal(dp[1]))
                                    d_dep = abs(cx - dp[0]) / DEPTH_PX_PER_YD
                                    d = math.hypot(d_lat, d_dep)
                                else:
                                    d = math.hypot(cx - dp[0], cy - dp[1])
                                if best_open is None or d < best_open:
                                    best_open = d
                            if best_open is not None:
                                unit_str = "yd" if lateral_cal else "px"
                                o_lbl = f"O:{best_open:.1f}{unit_str}"
                                coord_label = (coord_label + " " + o_lbl
                                               if coord_label else o_lbl)
                # Check for merged yellow+blue box (defender pressed against receiver)
                # Only split when BOTH halves classify cleanly - one OFFENSE, one DEFENSE.
                # Otherwise treat as a single OFFENSE box and skip the split.  This
                # prevents jersey/pants colour noise on a lone receiver from
                # producing a phantom DEFENSE sliver labelled with the receiver's own ID.
                bbox_crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None
                split_x   = find_horizontal_color_split(bbox_crop) if bbox_crop is not None else None
                do_split = False
                if split_x is not None and not is_oob:
                    abs_split = x1 + split_x
                    lt, lj, ls, _ = classify_player(frame[y1:y2, x1:abs_split])
                    rt, rj, rs, _ = classify_player(frame[y1:y2, abs_split:x2])
                    # Require one side OFFENSE and the other DEFENSE - no UNKNOWNs allowed.
                    do_split = ({lt, rt} == {"OFFENSE", "DEFENSE"})
                if do_split:
                    tracked_is_left = (lt == final_team or
                                       (final_team == "QB" and lt == "OFFENSE"))
                    if tracked_is_left:
                        draw_box(frame, x1, y1, abs_split, y2, eff, final_team, lj, ls,
                                 role=role, coord_label=coord_label)
                        draw_box(frame, abs_split, y1, x2, y2, eff, rt, rj, rs)
                    else:
                        draw_box(frame, x1, y1, abs_split, y2, eff, lt, lj, ls)
                        draw_box(frame, abs_split, y1, x2, y2, eff, final_team, rj, rs,
                                 role=role, coord_label=coord_label)
                else:
                    draw_box(frame,x1,y1,x2,y2,eff,final_team,jc,shoe,
                             role=role, coord_label=coord_label, dim=is_oob)
            # Ball
            for (bx1,by1,bx2,by2,bc) in ball_dets.get(fi,[]):
                cv2.rectangle(frame,(bx1,by1),(bx2,by2),BALL_COL,2)
                cv2.putText(frame,f"ball {bc:.2f}",(bx1,by1-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,BALL_COL,1)
            # Snap / throw banners
            if snap_frame is not None:
                if fi == snap_frame:
                    draw_banner(frame,">>> SNAP <<<",SNAP_COL,row=0)
                elif fi > snap_frame:
                    draw_banner(frame,
                        f"t+{(fi-snap_frame)/fps:.2f}s",SNAP_COL,row=0)
            if throw_frame is not None and fi >= throw_frame:
                draw_banner(frame,
                    ">>> THROW <<<" if fi==throw_frame else "post-throw",
                    THROW_COL,row=1)
            # OL highlight at snap frame only
            if snap_frame is not None and fi == snap_frame:
                for t in sl_tracks.values():
                    if t["id"] in ol_ids and fi in t["bboxes"]:
                        x1b,y1b,x2b,y2b = t["bboxes"][fi]
                        cv2.rectangle(frame,(x1b-3,y1b-3),(x2b+3,y2b+3),
                                      COL["OL"],3)
            # Ghost defender positions: draw estimated centroid for defenders
            # that are not detected in this frame but have a ghost position.
            # Drawn as a dim dashed circle so it's visually distinct from a
            # real detection box.
            # Collect real defender centroids this frame (any track ID)
            real_def_centroids = []
            real_def_ids = set()
            for (tid,x1,y1,x2,y2,*_) in sl_dets.get(fi, []):
                eff = track_merge_map.get(tid, tid)
                if sl_tracks.get(eff, {}).get("team") == "DEFENSE":
                    real_def_ids.add(eff)
                    real_def_centroids.append(((x1+x2)/2, (y1+y2)/2))

            GHOST_SUPPRESS_PX = 80   # suppress ghost if any real defender is this close
            GHOST_DEDUP_PX    = 60   # suppress second ghost if two are this close
            drawn_ghost_pos   = []   # (cx,cy) of ghost circles already drawn this frame
            for def_id, ghost in defender_ghosts.items():
                if def_id in real_def_ids: continue      # same track already drawn
                if snap_frame is None or fi <= snap_frame: continue   # no ghosts pre-snap
                if fi not in ghost: continue
                # Only draw if this frame was flagged as a display frame -
                # suppresses short-gap flicker ghosts and no-context direction ghosts
                if fi not in ghost_display_frames.get(def_id, set()): continue
                gcx, gcy = ghost[fi]
                # Suppress if a real defender is already nearby (re-ID cover)
                if any(math.hypot(gcx - rcx, gcy - rcy) < GHOST_SUPPRESS_PX
                       for rcx, rcy in real_def_centroids):
                    continue
                # Suppress if another ghost circle is already drawn nearby
                if any(math.hypot(gcx - px, gcy - py) < GHOST_DEDUP_PX
                       for px, py in drawn_ghost_pos):
                    continue
                gx, gy = int(gcx), int(gcy)
                cv2.circle(frame, (gx, gy), 14, (140, 50, 0), 1)
                cv2.circle(frame, (gx, gy), 3,  (140, 50, 0), -1)
                cv2.putText(frame, f"#{def_id}?",
                            (gx + 8, gy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 50, 0), 1,
                            cv2.LINE_AA)
                drawn_ghost_pos.append((gcx, gcy))

        elif in_ez_out:
            draw_banner(frame, "EZ ZOOMED-OUT (skipped)", (80,80,80), row=0)

        cv2.putText(frame,
            f"f{fi} [{'SL' if in_sl else 'EZ-in' if in_ez_in else 'EZ-out'}]",
            (W-220,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
        writer.write(frame)
        if fi % 300 == 0: print(f"  rendered {fi}/{total}")

    cap_r.release()
    writer.release()
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
