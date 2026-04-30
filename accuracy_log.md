# Triangulation Accuracy Log

Benchmark: 4-image set (hawk_hill, ggb3, marina, angel_island), manually clicked pixels.  
Ground truth: GGB midspan (37.8199°N, 122.4783°W).  
Run: `conda run -n triangulator python3 baseline.py`

| Date       | Method                        | Error (m) | Notes                                        |
|------------|-------------------------------|----------:|----------------------------------------------|
| 2026-04-29 | Pairwise intersection average |       580 | Original — averaged N*(N-1)/2 ray pairs      |
| 2026-04-29 | Least-squares (perpendicular) |       122 | Closed-form LS: P* = (Σ Aᵢ)⁻¹ (Σ Aᵢ pᵢ)   |
| 2026-04-29 | LS + approximate calibration  |       169 | cv2.undistortPoints with estimated K & dist — worse because distortion coefficients are guesses; run calibrate.py with checkerboard to improve |
| 2026-04-29 | LS + measured calibration     |       131 | ChArUco calibration: rear 30 imgs RMS 0.97px, front 18 imgs RMS 0.36px |
| 2026-04-29 | LS + refined calibration      |       111 | Rear re-calibrated with 34 curated imgs, RMS 0.86px — new best         |
