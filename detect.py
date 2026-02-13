import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from detect_blocks import detect_blocks
from detect_board_cells import BoardDetectionResult, detect_board_cells
from detect_energy_bars import detect_energy_bars
from solver import Block, Puzzle, save_solution_image, solve_puzzle


def hue_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 180.0 - d)


def hsv_hue_to_rgb(hue: float) -> List[int]:
    h = int(round(hue)) % 180
    hsv = np.uint8([[[h, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return [int(bgr[2]), int(bgr[1]), int(bgr[0])]


def inner_cell_hue(
    warped_bgr: np.ndarray,
    rows: int,
    cols: int,
    row: int,
    col: int,
    margin_ratio: float = 0.20,
) -> float:
    h, w = warped_bgr.shape[:2]
    cw = w / cols
    ch = h / rows
    x1 = int(round(col * cw))
    x2 = int(round((col + 1) * cw))
    y1 = int(round(row * ch))
    y2 = int(round((row + 1) * ch))
    cell = warped_bgr[y1:y2, x1:x2]
    if cell.size == 0:
        return 0.0

    hh, ww = cell.shape[:2]
    mx = min(ww // 3, max(1, int(round(ww * margin_ratio))))
    my = min(hh // 3, max(1, int(round(hh * margin_ratio))))
    cx1 = min(ww - 1, mx)
    cy1 = min(hh - 1, my)
    cx2 = max(cx1 + 1, ww - mx)
    cy2 = max(cy1 + 1, hh - my)
    patch = cell[cy1:cy2, cx1:cx2]
    if patch.size == 0:
        patch = cell

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 40) & (val > 35)
    if int(mask.sum()) == 0:
        return float(np.median(hsv[:, :, 0]))
    return float(np.median(hsv[:, :, 0][mask]))


def build_lock_matrices(
    board: BoardDetectionResult,
    color_centers: List[float],
) -> Dict[str, List[List[int]]]:
    rows, cols = board.rows, board.cols
    color_count = max(1, len(color_centers))
    out: Dict[str, List[List[int]]] = {}
    for i in range(color_count):
        out[f"colour_{i+1}"] = [[0 for _ in range(cols)] for _ in range(rows)]

    if color_count == 1:
        mat = out["colour_1"]
        for r in range(rows):
            for c in range(cols):
                if board.lock_matrix[r][c] == 1:
                    mat[r][c] = 1
        return out

    for r in range(rows):
        for c in range(cols):
            if board.lock_matrix[r][c] != 1:
                continue
            hue = inner_cell_hue(board.warped_board, rows, cols, r, c)
            dists = [hue_dist(hue, hc) for hc in color_centers]
            ci = int(np.argmin(dists))
            out[f"colour_{ci+1}"][r][c] = 1
    return out


def matrix_sum(mat: List[List[int]]) -> int:
    return int(sum(sum(int(v) for v in row) for row in mat))


def shape_area(shape: List[List[int]]) -> int:
    return int(sum(sum(int(v) for v in row) for row in shape))


def infer_board_cell_size(corners: Dict[str, Tuple[int, int]], rows: int, cols: int) -> float:
    tl = np.array(corners["tl"], dtype=np.float32)
    tr = np.array(corners["tr"], dtype=np.float32)
    br = np.array(corners["br"], dtype=np.float32)
    bl = np.array(corners["bl"], dtype=np.float32)
    board_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    board_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(0.5 * ((board_w / max(1, cols)) + (board_h / max(1, rows))))


def trim_shape(shape: List[List[int]]) -> List[List[int]]:
    if not shape or not shape[0]:
        return shape
    arr = np.array(shape, dtype=np.uint8)
    rs = np.where(arr.sum(axis=1) > 0)[0]
    cs = np.where(arr.sum(axis=0) > 0)[0]
    if len(rs) == 0 or len(cs) == 0:
        return shape
    out = arr[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1]
    return out.astype(int).tolist()


def shape_to_cells(shape: List[List[int]]) -> Set[Tuple[int, int]]:
    cells: Set[Tuple[int, int]] = set()
    for r, row in enumerate(shape):
        for c, v in enumerate(row):
            if int(v) == 1:
                cells.add((r, c))
    return cells


def cells_to_shape(cells: Set[Tuple[int, int]]) -> List[List[int]]:
    if not cells:
        return [[1, 1], [1, 1]]
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in norm:
        out[r][c] = 1
    return out


def cells_connected(cells: Set[Tuple[int, int]]) -> bool:
    if not cells:
        return False
    q: deque[Tuple[int, int]] = deque()
    start = next(iter(cells))
    q.append(start)
    seen: Set[Tuple[int, int]] = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def shape_key(shape: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in trim_shape(shape))


def line_shape(length: int, horizontal: bool) -> List[List[int]]:
    if horizontal:
        return [[1 for _ in range(length)]]
    return [[1] for _ in range(length)]


def shape_style_penalty(shape: List[List[int]]) -> int:
    # For tall, narrow pieces, a row pattern [0,1] is usually a contour artifact.
    h = len(shape)
    w = len(shape[0]) if h else 0
    if w == 2 and h >= 4:
        bad = 0
        for row in shape:
            if row[0] == 0 and row[1] == 1:
                bad += 1
        return bad
    return 0


@dataclass
class BlockCandidate:
    blob_id: str
    color_idx: int
    hue_med: float
    hue_diff: float
    bbox: Tuple[int, int, int, int]
    shape: List[List[int]]


@dataclass
class ShapeVariant:
    shape: List[List[int]]
    area: int
    penalty: int


def generate_shape_variants(shape: List[List[int]], bbox: Tuple[int, int, int, int]) -> List[ShapeVariant]:
    base = trim_shape(shape)
    cells = shape_to_cells(base)
    variants: Dict[Tuple[Tuple[int, ...], ...], ShapeVariant] = {}

    def add_variant(s: List[List[int]], penalty: int) -> None:
        ss = trim_shape(s)
        k = shape_key(ss)
        ar = shape_area(ss)
        if ar < 2:
            return
        p = int(penalty + shape_style_penalty(ss))
        old = variants.get(k)
        if old is None or p < old.penalty:
            variants[k] = ShapeVariant(shape=ss, area=ar, penalty=p)

    add_variant(base, penalty=0)
    x, y, w, h = bbox
    del x, y
    aspect = float(w) / max(1.0, float(h))
    ar = shape_area(base)

    # If a tiny bar-like shape was underestimated, add line alternatives.
    if ar <= 2 and aspect >= 2.8:
        for L in (3, 4):
            add_variant(line_shape(L, horizontal=True), penalty=1)
    if ar <= 2 and (1.0 / max(0.001, aspect)) >= 2.8:
        for L in (3, 4):
            add_variant(line_shape(L, horizontal=False), penalty=1)

    # For small contour errors, allow one-cell removal while keeping connectivity.
    if ar >= 4:
        for cell in list(cells):
            reduced = set(cells)
            reduced.remove(cell)
            if len(reduced) < 2 or not cells_connected(reduced):
                continue
            add_variant(cells_to_shape(reduced), penalty=1)

    return sorted(variants.values(), key=lambda v: (v.penalty, -v.area))


def pick_blob_color_idx(blob: Dict[str, object], color_centers: List[float]) -> int:
    if color_centers and "hue_median" in blob:
        hue = float(blob["hue_median"])
        dists = [hue_dist(hue, hc) for hc in color_centers]
        return int(np.argmin(dists))

    raw = str(blob.get("colour", "colour_1"))
    if raw.startswith("colour_"):
        try:
            return max(0, int(raw.split("_")[1]) - 1)
        except Exception:
            pass
    return 0


def payload_to_puzzle(payload: Dict[str, object]) -> Puzzle:
    board = payload["board"]  # type: ignore[assignment]
    gray = payload["gray_cells"]  # type: ignore[assignment]
    locks = payload["locks"]  # type: ignore[assignment]
    cmap = payload.get("color_map", {})  # type: ignore[assignment]

    row_targets: Dict[int, List[int]] = {}
    col_targets: Dict[int, List[int]] = {}
    for ck, vv in board.items():
        ci = int(str(ck).split("_")[1])
        col_targets[ci] = [int(x) for x in vv[0]]
        row_targets[ci] = [int(x) for x in vv[1]]

    rows = len(gray)
    cols = len(gray[0]) if rows else 0
    locks_out: Dict[int, List[List[int]]] = {}
    for ck, mat in locks.items():
        ci = int(str(ck).split("_")[1])
        locks_out[ci] = [[int(v) for v in row] for row in mat]
    for ci in row_targets.keys():
        if ci not in locks_out:
            locks_out[ci] = [[0 for _ in range(cols)] for _ in range(rows)]

    color_rgb: Dict[int, Tuple[int, int, int]] = {}
    if isinstance(cmap, dict):
        for ck, rgb in cmap.items():
            ci = int(str(ck).split("_")[1])
            color_rgb[ci] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    blocks: List[Block] = []
    for k in sorted(payload.keys()):
        if not str(k).startswith("block_"):
            continue
        b = payload[k]
        color_val = b["colour"]
        if isinstance(color_val, str):
            ci = int(color_val.split("_")[1])
        else:
            ci = int(color_val)
        shp = [[int(v) for v in row] for row in b["shape"]]
        blocks.append(Block(block_id=str(k), color=ci, shape=shp))

    return Puzzle(
        rows=rows,
        cols=cols,
        row_targets=row_targets,
        col_targets=col_targets,
        gray=[[int(v) for v in row] for row in gray],
        locks=locks_out,
        blocks=blocks,
        color_rgb=color_rgb,
    )


def required_area_by_color(
    board_obj: Dict[str, List[List[int]]],
    locks_obj: Dict[str, List[List[int]]],
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for ck, vv in board_obj.items():
        ci = int(ck.split("_")[1])
        target = int(sum(int(x) for x in vv[1]))
        lock_cnt = matrix_sum(locks_obj[ck])
        out[ci] = max(0, target - lock_cnt)
    return out


def build_candidate_blocks(
    blocks: Dict[str, object],
    color_centers: List[float],
    cell_size: float,
) -> List[BlockCandidate]:
    raw = list(blocks.get("blobs", []))
    out: List[BlockCandidate] = []
    for b in raw:
        shp = b.get("shape")
        if not isinstance(shp, list) or not shp:
            continue
        ci = pick_blob_color_idx(b, color_centers)
        hue = float(b.get("hue_median", color_centers[min(ci, len(color_centers) - 1)] if color_centers else 35.0))
        hd = 0.0
        if color_centers:
            hd = hue_dist(hue, color_centers[min(ci, len(color_centers) - 1)])
        bx, by, bw, bh = [int(v) for v in b["bbox_image"]]
        # Remove obvious noise blobs.
        if hd > 18.0:
            continue
        if max(bw, bh) < max(10.0, 0.32 * cell_size):
            continue
        out.append(
            BlockCandidate(
                blob_id=str(b.get("blob_id", "blob")),
                color_idx=ci,
                hue_med=hue,
                hue_diff=hd,
                bbox=(bx, by, bw, bh),
                shape=trim_shape(shp),
            )
        )
    return out


def solve_with_block_variants(
    base_payload: Dict[str, object],
    candidates: List[BlockCandidate],
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    board_obj = base_payload["board"]  # type: ignore[assignment]
    locks_obj = base_payload["locks"]  # type: ignore[assignment]
    req = required_area_by_color(board_obj, locks_obj)
    colors = sorted(req.keys())
    if not candidates:
        return None, None

    variants_per_block: List[List[ShapeVariant]] = []
    color_ids: List[int] = []
    for b in candidates:
        vars0 = generate_shape_variants(b.shape, b.bbox)
        # Optional drop for low-confidence tiny block.
        weak = b.hue_diff > 12.0 or shape_area(b.shape) <= 2
        if weak:
            vars0.append(ShapeVariant(shape=[], area=0, penalty=3))
        variants_per_block.append(vars0)
        color_ids.append(b.color_idx + 1)

    best_payload: Optional[Dict[str, object]] = None
    best_solution: Optional[Dict[str, object]] = None
    best_penalty: Optional[int] = None
    N = len(candidates)
    area_now = {c: 0 for c in colors}
    chosen_idx = [0 for _ in range(N)]

    # Precompute max remaining area per color for pruning.
    max_remain: List[Dict[int, int]] = []
    rem = {c: 0 for c in colors}
    for i in range(N - 1, -1, -1):
        c = color_ids[i]
        rem[c] += max(v.area for v in variants_per_block[i])
        max_remain.append(dict(rem))
    max_remain = list(reversed(max_remain))

    def dfs(i: int, penalty: int) -> None:
        nonlocal best_payload, best_solution, best_penalty
        if best_penalty is not None and penalty > best_penalty:
            return
        if i == N:
            for c in colors:
                if area_now[c] != req[c]:
                    return
            payload = {
                "color_map": base_payload["color_map"],
                "board": base_payload["board"],
                "gray_cells": base_payload["gray_cells"],
                "locks": base_payload["locks"],
            }
            bi = 1
            for k in range(N):
                v = variants_per_block[k][chosen_idx[k]]
                if v.area <= 0:
                    continue
                payload[f"block_{bi:03d}"] = {"colour": f"colour_{color_ids[k]}", "shape": v.shape}
                bi += 1
            puzzle = payload_to_puzzle(payload)
            try:
                solved = solve_puzzle(puzzle)
            except ValueError:
                solved = None
            if solved is None:
                return
            best_penalty = penalty
            best_payload = payload
            best_solution = solved
            return

        c = color_ids[i]
        for vi, var in enumerate(variants_per_block[i]):
            nxt = area_now[c] + var.area
            if nxt > req[c]:
                continue
            # Remaining max area cannot reach target.
            remain = max_remain[i + 1][c] if i + 1 < N else 0
            if nxt + remain < req[c]:
                continue
            area_now[c] = nxt
            chosen_idx[i] = vi
            dfs(i + 1, penalty + var.penalty)
            area_now[c] -= var.area

    dfs(0, 0)
    return best_payload, best_solution


def copy_payload(payload: Dict[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(payload))


def solve_payload(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    try:
        return solve_puzzle(payload_to_puzzle(payload))
    except ValueError:
        return None


def solve_with_gray_relaxation(
    base_payload: Dict[str, object],
    max_flips: int = 2,
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    gray = base_payload["gray_cells"]  # type: ignore[assignment]
    locks = base_payload["locks"]  # type: ignore[assignment]
    rows = len(gray)
    cols = len(gray[0]) if rows else 0

    lock_any = [[0 for _ in range(cols)] for _ in range(rows)]
    for mat in locks.values():
        for r in range(rows):
            for c in range(cols):
                if int(mat[r][c]) == 1:
                    lock_any[r][c] = 1

    candidates: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if int(gray[r][c]) == 1 and lock_any[r][c] == 0:
                candidates.append((r, c))
    if not candidates:
        return None, None

    # Prefer cells near the board center; corner false-positives are rarer.
    rc = (rows - 1) * 0.5
    cc = (cols - 1) * 0.5
    candidates.sort(key=lambda p: abs(p[0] - rc) + abs(p[1] - cc))
    candidates = candidates[:12]

    def test_flip(flip_list: List[Tuple[int, int]]) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        payload = copy_payload(base_payload)
        g = payload["gray_cells"]  # type: ignore[assignment]
        for rr, cc2 in flip_list:
            g[rr][cc2] = 0
        solved = solve_payload(payload)
        if solved is not None:
            return payload, solved
        return None, None

    for i in range(len(candidates)):
        p, s = test_flip([candidates[i]])
        if p is not None:
            return p, s

    if max_flips < 2:
        return None, None

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            p, s = test_flip([candidates[i], candidates[j]])
            if p is not None:
                return p, s
    return None, None


def build_standard_puzzle_json(
    board: BoardDetectionResult,
    energy: Dict[str, object],
    blocks: Dict[str, object],
) -> Dict[str, object]:
    rows, cols = board.rows, board.cols
    erows = int(energy["rows"])
    ecols = int(energy["cols"])
    if rows != erows or cols != ecols:
        raise ValueError(f"Board-size mismatch: board_cells={rows}x{cols}, energy_bars={erows}x{ecols}")

    color_count = int(energy["color_count"])
    color_centers = [float(v) for v in energy.get("color_hue_centers", [])]
    if len(color_centers) < color_count:
        if color_centers:
            while len(color_centers) < color_count:
                color_centers.append(color_centers[-1])
        else:
            color_centers = [35.0 for _ in range(color_count)]
    else:
        color_centers = color_centers[:color_count]

    board_obj: Dict[str, List[List[int]]] = {}
    col_targets: Dict[str, List[int]] = energy["col_targets"]  # type: ignore[assignment]
    row_targets: Dict[str, List[int]] = energy["row_targets"]  # type: ignore[assignment]
    for i in range(color_count):
        ck = f"colour_{i+1}"
        board_obj[ck] = [list(col_targets[ck]), list(row_targets[ck])]

    locks_obj = build_lock_matrices(board, color_centers)
    colors_obj = {f"colour_{i+1}": hsv_hue_to_rgb(color_centers[i]) for i in range(color_count)}

    payload: Dict[str, object] = {
        "color_map": colors_obj,
        "board": board_obj,
        "gray_cells": board.gray_matrix,
        "locks": locks_obj,
    }

    cell_size = infer_board_cell_size(board.corners, board.rows, board.cols)
    selected = build_candidate_blocks(blocks, color_centers, cell_size)
    for i, b in enumerate(selected, start=1):
        payload[f"block_{i:03d}"] = {
            "colour": f"colour_{b.color_idx + 1}",
            "shape": b.shape,
        }
    return payload


def solution_payload(result: Dict[str, object]) -> Dict[str, object]:
    occupancy = result["occupancy"]
    block_grid = result["block_grid"]
    placements_raw = result["placements"]

    placements: List[Dict[str, object]] = []
    for r0, c0, ob in placements_raw:
        cells = [[int(r0 + dr), int(c0 + dc)] for dr, dc in ob.cells]
        placements.append(
            {
                "block_id": ob.block_id,
                "colour": f"colour_{ob.color}",
                "origin": [int(r0), int(c0)],
                "cells": cells,
            }
        )

    return {
        "occupancy": occupancy,
        "block_grid": block_grid,
        "placements": placements,
    }


def default_paths(input_image: str) -> Tuple[str, str, str]:
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_image))[0]
    puzzle_json = os.path.join("outputs", f"{base}_puzzle.json")
    solution_png = os.path.join("outputs", f"{base}_solution.png")
    solution_json = os.path.join("outputs", f"{base}_solution.json")
    return puzzle_json, solution_png, solution_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect puzzle data from screenshot, build standard JSON, and solve with solver.py."
    )
    parser.add_argument("input", help="Path to screenshot image")
    parser.add_argument("--templates", default="templates", help="Path to templates directory")
    parser.add_argument("--json-out", default=None, help="Output path for standard puzzle JSON")
    parser.add_argument("--solution-out", default=None, help="Output path for solution image (PNG)")
    parser.add_argument("--solution-json-out", default=None, help="Output path for solution JSON")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {args.input}")

    puzzle_json_default, solution_png_default, solution_json_default = default_paths(args.input)
    puzzle_json_path = args.json_out or puzzle_json_default
    solution_png_path = args.solution_out or solution_png_default
    solution_json_path = args.solution_json_out or solution_json_default

    for p in (puzzle_json_path, solution_png_path, solution_json_path):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    board = detect_board_cells(img_bgr, args.templates)
    energy = detect_energy_bars(img_bgr, args.templates, board=board)  # type: ignore[assignment]
    blocks, _ = detect_blocks(img_bgr, args.templates, board=board, energy=energy)

    payload = build_standard_puzzle_json(board, energy, blocks)
    result = solve_payload(payload)

    if result is None:
        centers = [float(v) for v in energy.get("color_hue_centers", [])]
        cands = build_candidate_blocks(blocks, centers, infer_board_cell_size(board.corners, board.rows, board.cols))
        alt_payload, alt_result = solve_with_block_variants(payload, cands)
        if alt_payload is not None and alt_result is not None:
            payload = alt_payload
            result = alt_result
        else:
            gray_payload, gray_result = solve_with_gray_relaxation(payload, max_flips=2)
            if gray_payload is not None and gray_result is not None:
                payload = gray_payload
                result = gray_result
            else:
                with open(puzzle_json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print("No solution found.")
                return 1

    with open(puzzle_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    puzzle = payload_to_puzzle(payload)
    save_solution_image(puzzle, result["occupancy"], result["block_grid"], solution_png_path)
    with open(solution_json_path, "w", encoding="utf-8") as f:
        json.dump(solution_payload(result), f, ensure_ascii=False, indent=2)

    print(f"Saved standard puzzle JSON: {puzzle_json_path}")
    print(f"Saved solution image: {solution_png_path}")
    print(f"Saved solution JSON: {solution_json_path}")
    print(f"Detected board size: {board.rows} x {board.cols}")
    print(f"Detected colors: {energy['color_count']}")
    print(f"Detected blocks: {blocks.get('detected_blob_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
