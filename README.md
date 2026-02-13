# Arknights Endfield Puzzle Detector and Solver

This project reads an Arknights: Endfield puzzle screenshot, detects all puzzle elements, builds a standard puzzle JSON, and solves it with `solver.py`.

End-to-end flow:
1. Input screenshot.
2. Detect board tags, board cells (`empty/gray/lock`), energy bars, and right-side blocks.
3. Build standard puzzle JSON.
4. Solve and export solution files.

## Project Layout

- `detect.py`: one-shot entrypoint (`screenshot -> puzzle JSON -> solve`).
- `detect_board_tags.py`: detect 4 board corner tags.
- `detect_board_cells.py`: classify board cells.
- `detect_energy_bars.py`: detect top and left energy bars by color.
- `detect_blocks.py`: detect right-side colored block regions and infer shapes.
- `solver.py`: parse JSON and solve by backtracking.
- `templates/`: template assets.
- `examples_json/`: reference JSON format samples.
- `pic_examples/`: sample screenshots and block image references.
- `outputs/`: generated outputs.

## Requirements

- Python 3.10+ (3.11 recommended)
- OpenCV (`cv2`)
- NumPy
- Matplotlib

Recommended environment:

```powershell
conda activate opencv-env
```

## Quick Start

Run full detection and solving:

```powershell
py detect.py <image_path> --templates templates
```

Example:

```powershell
py detect.py actual_07.png --templates templates
```

Default outputs:

- `outputs/<name>_puzzle.json`
- `outputs/<name>_solution.png`
- `outputs/<name>_solution.json`

## Module Commands

Detect board tags:

```powershell
py detect_board_tags.py <image_path> --templates templates --out outputs/<name>_tags.png
```

Detect board cells:

```powershell
py detect_board_cells.py <image_path> --templates templates
```

Detect board cells with confidence debug:

```powershell
py detect_board_cells.py <image_path> --templates templates --save-debug
```

Extra debug outputs for `--save-debug`:

- `outputs/<name>_board_cells_conf_debug.png`
- `outputs/<name>_board_cells_debug.json`

Detect energy bars:

```powershell
py detect_energy_bars.py <image_path> --templates templates
```

Detect blocks with debug images:

```powershell
py detect_blocks.py <image_path> --templates templates --save-debug
```

Debug outputs from `detect_blocks.py --save-debug`:

- `*_blocks_debug.png`
- `*_linefit_debug.png`
- `*_contour_debug.png`
- `*_grid_debug.png`
- `*_right_roi.png`
- `*_right_roi_mask.png`

## Current Detection Rules (Important)

Board cells (`detect_board_cells.py`):

- Strong prior for `gray`: low saturation + medium brightness.
- Strong prior for `lock`: high saturation + high brightness.
- If priors do not trigger, template score fallback is used.
- `gray` in fallback path must pass confidence gate; otherwise it falls back to `empty` (precision-first behavior).
- Very low-confidence cells default to `empty`.

Blocks (`detect_blocks.py`):

- Candidate region hard constraints:
- `12 <= w <= 90`
- `12 <= h <= 90`
- `w * h >= 900` (new region-area filter)
- Candidate must pass color validity (`min_valid_pixels`) and board-side position filters.
- Shape inference uses axis-aligned contour/line fitting and multi-candidate grid scoring.

## Standard Puzzle JSON Format

`detect.py` outputs:

```json
{
  "color_map": {
    "colour_1": [170, 230, 32],
    "colour_2": [32, 124, 230]
  },
  "board": {
    "colour_1": [[2, 1, 4], [3, 2, 2]],
    "colour_2": [[1, 3, 0], [0, 2, 2]]
  },
  "gray_cells": [[0, 1], [0, 0]],
  "locks": {
    "colour_1": [[0, 0], [0, 0]],
    "colour_2": [[0, 1], [0, 0]]
  },
  "block_001": {"colour": "colour_1", "shape": [[1, 1], [1, 0]]},
  "block_002": {"colour": "colour_2", "shape": [[1, 1, 1]]}
}
```

Notes:

- `board.colour_n[0]`: column targets (top bars).
- `board.colour_n[1]`: row targets (left bars).
- `gray_cells`: unavailable cells.
- `locks.colour_n`: locked cells per color.
- `block_xxx.shape`: binary matrix.

## Known Status

- Core examples in `pic_examples` are supported for current pipeline.
- Some real screenshots may still fail due to upstream detection noise (bars or colors).
- `detect.py` contains repair logic for inconsistent targets, but it is still fallback behavior.
