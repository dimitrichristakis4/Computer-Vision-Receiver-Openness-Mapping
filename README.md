# Computer-Vision-Receiver-Openness-Mapping
A computer vision pipeline that tracks receiver openness in yards on football film. Built on Queen's Varsity Football practice footage using YOLOv8 and ByteTrack.

**Status:** Functional on the specific clip used for development. Has not been tested across different footage conditions - lighting, camera angle, jersey colors, and field markings will all affect performance and likely require recalibration. This is a working proof of concept, not a generalized tool.

## What it does

Processes raw game film and outputs a per-play openness score for each receiver at the moment of the throw, plus a full time series of separation from snap to throw.

Pipeline steps:
- Hard cut detection to isolate individual plays
- YOLOv8 player detection + ByteTrack multi-object tracking
- HSV color classification to separate offense from defense
- Two-axis yard calibration: depth uses a fixed pixels-per-yard constant (linear from the elevated sideline angle); lateral uses known field landmarks (near sideline, hash marks, far sideline) to interpolate real-world position and correct for perspective compression on the far side of the field
- Snap detection via player stillness before sudden movement
- Receiver numbering by field position (outermost to boundary = R1)
- Openness calculated as lateral + depth separation from nearest defender, in yards
- Openness gated from 1s post-snap - before that reflects formation alignment, not route separation
- Throw frame detected via optical flow spike

## How openness is calculated

At every frame, each receiver gets an openness score: the straight-line distance in yards to their nearest defender. No direction or coverage type is factored in - just who is closest and how far away they are.

Getting that distance in real yards requires converting pixel positions to field coordinates. Depth (downfield) uses a fixed pixels-per-yard constant because the elevated sideline camera captures downfield movement linearly with no perspective distortion. Lateral (sideline to sideline) is harder - the far side of the field is compressed in pixel space, so the pipeline detects known CFL field landmarks (near sideline at 0 yards, near hash at 24, far hash at 41, far sideline at 65) and interpolates each player's real lateral position from wherever they fall between those anchors.

Openness is only tracked from 1 second after the snap because before that players are still in formation, not running routes.

## Outputs

- `openness_results.csv` - max receiver openness per play at the moment of the throw
- `openness_timeseries.csv` - full openness time series for every receiver from snap to throw
- `openness_chart.png` - visualization of receiver separation over time

Run `plot_openness.py` to regenerate the chart from the time series CSV.

## Usage

```bash
pip install -r requirements.txt
python receiver_openness_pipeline.py
```

You will need to supply your own film clips. The pipeline expects `.mov` files named by day and play number. Update the file paths at the top of `receiver_openness_pipeline.py` to point to your footage.

The YOLOv8 model weight (`yolov8m.pt`) is not included. Download it via:

```python
from ultralytics import YOLO
YOLO('yolov8m.pt')
```

## Notes

Tested on Apple Silicon (M4) with MPS acceleration. On other hardware, PyTorch will fall back to CPU automatically.

Team classification uses HSV color thresholds tuned to Queen's red and the opposing team's colors from the specific film used. If applying to different footage, the HSV ranges in `receiver_openness_pipeline.py` will need to be recalibrated for the teams in frame.
