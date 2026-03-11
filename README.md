# NutriVision

Food photo to nutrition estimator built across Python, C++, FastAPI, ONNX Runtime, and React.

Current state:
- Project scaffold is in place.
- Source files exist as learning-oriented skeletons.
- Core implementation is still intentionally unfinished.

Suggested build order:
1. `data/` for dataset download and preprocessing.
2. `model/` for training and ONNX export.
3. `inference/` for C++ runtime integration.
4. `api/` for serving predictions and nutrition lookups.
5. `frontend/` for the upload and results experience.
