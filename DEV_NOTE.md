# DEV_NOTE

Maintenance notes for the current detector and solver implementation.

## 1. End-to-End Flow

Entrypoint: `detect.py`

Pipeline:
1. `detect_board_cells.detect_board_cells()`: board size, corners, `gray_matrix`, `lock_matrix`.
2. `detect_energy_bars.detect_energy_bars()`: per-color top/left targets.
3. `detect_blocks.detect_blocks()`: right-side blocks (`bbox`, color, shape).
4. `detect.py.build_standard_puzzle_json()`: normalize into solver JSON.
5. `solver.parse_puzzle()` + `solver.solve_puzzle()`.
6. `solver.save_solution_image()`.

## 2. Module Contracts

### 2.1 `detect_board_tags.py`

- Input: raw screenshot (BGR).
- Preprocess: resize to canonical `1709 x 961`.
- Output: four board corners (`tl/tr/br/bl`).
- Strategy: center-out candidate search and non-greedy final combination scoring.

### 2.2 `detect_board_cells.py`

- Input: screenshot + `templates/board_*.png`.
- Output: `BoardDetectionResult` with `rows`, `cols`, `corners`, `warped_board`, `cell_types`, `gray_matrix`, `lock_matrix`.
- Grid size inference: board geometry divided by template-like cell size.

Current classification behavior:
- Strong priors:
- `lock`: high saturation + high brightness.
- `gray`: very low saturation + medium brightness.
- Fallback:
- template score fallback is used if priors do not trigger.
- `gray` fallback requires confidence gate (score and margin + soft HSV bounds).
- if `gray` gate fails, fallback defaults to `empty` (precision-first).
- very low-confidence cells default to `empty`.

Debug support:
- `py detect_board_cells.py <img> --save-debug`
- Outputs:
- `*_board_cells_conf_debug.png`
- `*_board_cells_debug.json` (`scores`, `top/second`, `margin`, `rule_path`, crop info per cell)

### 2.3 `detect_energy_bars.py`

- Input: screenshot + bar templates.
- Output: `row_targets` and `col_targets` by `colour_n`.
- Strategy:
- local ROI extraction by row/column.
- contour/anchor sequence counting.
- hue clustering for color count (up to 2 colors).
- consistency balance between top/left totals.

### 2.4 `detect_blocks.py`

- Input: screenshot + board/energy context.
- Output includes `blobs[].bbox_image`, `blobs[].colour`, `blobs[].shape`, `blobs[].hue_median`, `blobs[].shape_debug`.

Current hard filters for candidate regions:
- `12 <= w <= 90`
- `12 <= h <= 90`
- `w * h >= 900` (small region reject before shape stage)
- minimum valid colored pixels filter (`min_valid_pixels`)
- board-side and y-range filters

Shape inference summary:
- local seam repair after component localization only.
- closed contour extraction + axis-aligned line fitting.
- minimum fitted segment length: `>= 9 px`.
- multi-candidate unit/grid scoring with anti-oversampling reduction.
- `shape_debug` stores `unit_candidates`, `grid_candidates`, `best_score`, `chosen_grid`.

Debug outputs from `--save-debug`:
- `*_blocks_debug.png`
- `*_linefit_debug.png`
- `*_contour_debug.png`
- `*_grid_debug.png`
- `*_right_roi.png`
- `*_right_roi_mask.png`

### 2.5 `detect.py`

- Main outputs:
- `<name>_puzzle.json`
- `<name>_solution.png`
- `<name>_solution.json`
- Important helpers:
- `build_lock_matrices()` assigns lock cells to colors in multi-color puzzles.
- `repair_targets_for_solver()` mitigates inconsistent bar totals under detection noise.

## 3. Regression Baseline

Use these as routine checks after behavior changes:
1. `pic_examples/single_color_01/single_color_01.png`
2. `pic_examples/single_color_02/single_color_02.png`
3. `pic_examples/double_color_01/double_color_01.png`
4. `actual_05.png` for board-cell gray false-positive checks.
5. `actual_07.png` for small-block rejection behavior (`w*h >= 900`).

## 4. Known Risks

1. Energy bars can still drift in low-contrast screenshots.
2. Multi-color hue separation is sensitive to lighting shifts.
3. `repair_targets_for_solver()` is fallback logic, not guaranteed true reconstruction.
4. Shape inference may still fail on heavily occluded or anti-aliased block borders.

## 5. Recommended Debug Order

1. `detect_board_tags.py`: check corner alignment first.
2. `detect_board_cells.py --save-debug`: confirm `E/G/L` and `rule_path`.
3. `detect_energy_bars.py`: validate row/column targets per color.
4. `detect_blocks.py --save-debug`: inspect bbox, contour, line-fit, grid score.
5. `detect.py`: final end-to-end verification.

## 6. Quick Commands

```powershell
conda activate opencv-env
py detect_board_cells.py actual_05.png --templates templates --save-debug
py detect_blocks.py actual_07.png --templates templates --save-debug
py detect.py actual_07.png --templates templates
py solver.py outputs/actual_07_puzzle.json
```

## 7. Change Guidelines

1. Keep module boundaries clean; avoid cross-module hidden coupling.
2. Any threshold change must keep or improve debug observability.
3. Prefer adding explainable gating over global morphology amplification.
4. Validate with regression screenshots before considering a change complete.
