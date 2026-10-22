# Weighbridge Truck Counter

**Counts trucks crossing a factory weighbridge in CCTV footage, and saves one timestamped screenshot per truck as audit evidence.**

Built at a textile manufacturing plant to check gate-pass vehicle counts against what the CCTV actually shows.

![python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![yolo](https://img.shields.io/badge/YOLO-Ultralytics-111F68)
![opencv](https://img.shields.io/badge/OpenCV-video-5C3EE8?logo=opencv)
![tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest)

## Why not just track and count line crossings?

The first version used multi-object tracking with direction-of-crossing counting. Real weighbridge footage has three things that make ID-based counting fragile:

- Trucks **reverse** off the bridge through the same camera view.
- The detector **flickers** on a stationary truck that's being weighed.
- A second lane behind the bridge carries trucks that never get weighed.

The fix uses the physics of the site. A weighbridge holds **one truck at a time**, so a short virtual line clipped to the bridge slab can act as an occupancy sensor with a debounced two-state machine:

```
CLEAR ──(line covered ≥ 1.0 s)──▶ COVERED   → count +1, save clearest frame
COVERED ──(line clear ≥ 2.5 s)──▶ CLEAR
```

Because the line is clipped to the slab, the back lane is ignored. The two time thresholds remove flicker, and a reversing truck re-covers the line before the clear timer finishes, so it isn't counted twice.

## Output

| File | Purpose |
|---|---|
| `counted.mp4` | Annotated video showing the line, boxes, live count and state |
| `shots/truckN_MMSS.jpg` | The **clean** frame (no overlays) with the highest-confidence detection for each truck, so the CCTV's burned-in clock stays readable |
| `truck_count.xlsx` | Proof sheet with columns #, video time, blank real-clock column, confidence and embedded screenshot. A reviewer fills in the real clock time and cross-checks it against weighbridge slips. |

## Usage

```bash
pip install -r requirements.txt
python -m truckcount.cli --video bridge.mp4 --line 596 134 362 --out output/
# --model best.pt  --classes 0 1    to use a custom-trained model
pytest -q tests
```

## Models

- **COCO `yolo11n`** (car/bus/truck) works out of the box for counting.
- **Custom empty/full truck detector.** YOLO trained on about 550 LabelMe-annotated frames from the site, reaching about 97–98% mAP50. It classifies whether a truck arrives loaded or empty. The weights and training data stay internal because they're site footage.

## Design notes

- `truckcount/line_counter.py` is pure Python with no dependencies, and the tests cover flicker, false positives and back-to-back trucks.
- Frames are processed with `vid_stride` (every 2nd frame by default). Thresholds are given in seconds and converted using the effective FPS, so changing the stride doesn't change behaviour.
- Ultralytics is imported lazily, so `--help` and the tests run without PyTorch.

## Limitations

- The approach assumes one vehicle on the slab at a time, which holds for a weighbridge but not for a general road.
- The line coordinates are per camera, so moving the camera means re-picking the line.
