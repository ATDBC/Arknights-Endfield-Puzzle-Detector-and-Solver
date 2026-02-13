import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detect_board_cells import BoardDetectionResult, detect_board_cells
from detect_energy_bars import detect_energy_bars


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


@dataclass
class ColorBlob:
    blob_id: str
    color_idx: int
    area: int
    hue_median: float
    sat_mean: float
    bbox_image: Rect
    shape: List[List[int]]
    unit_len: float


LineSeg = Tuple[int, int, int, int]
Contour = List[Tuple[int, int]]
MIN_FIT_SEG_LEN = 9


def hue_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 180.0 - d)


def infer_board_cell_size(corners: Dict[str, Point], rows: int, cols: int) -> float:
    tl = np.array(corners["tl"], dtype=np.float32)
    tr = np.array(corners["tr"], dtype=np.float32)
    br = np.array(corners["br"], dtype=np.float32)
    bl = np.array(corners["bl"], dtype=np.float32)
    board_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    board_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    cw = board_w / max(1, cols)
    ch = board_h / max(1, rows)
    return float(0.5 * (cw + ch))


def right_side_roi_from_board(
    img_shape: Tuple[int, int, int],
    corners: Dict[str, Point],
    cell_size: float,
) -> Rect:
    h, w = img_shape[:2]
    tr = corners["tr"]
    br = corners["br"]
    x_board_right = max(tr[0], br[0])
    x1 = max(0, int(round(x_board_right)))
    x2 = w
    return (x1, 0, x2, h)


def infer_color_centers_from_bars(
    img_bgr: np.ndarray,
    templates_dir: str,
    energy: Optional[Dict[str, object]] = None,
    board: Optional[BoardDetectionResult] = None,
) -> Optional[List[float]]:
    bars: Optional[Dict[str, object]] = energy
    if bars is None:
        try:
            bars = detect_energy_bars(img_bgr, templates_dir, board=board)  # type: ignore[assignment]
        except Exception:
            return None
    centers = bars.get("color_hue_centers")
    if not isinstance(centers, list) or not centers:
        return None
    vals: List[float] = []
    for c in centers[:2]:
        try:
            vals.append(float(c))
        except Exception:
            pass
    if not vals:
        return None
    vals.sort()
    return vals


def infer_color_centers_from_roi(roi_bgr: np.ndarray) -> List[float]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hch = hsv[:, :, 0]
    sch = hsv[:, :, 1]
    vch = hsv[:, :, 2]
    mask = (sch > 45) & (vch > 30)
    if int(mask.sum()) < 120:
        return [35.0]

    hist = np.zeros((180,), dtype=np.float32)
    hs = hch[mask].astype(np.int32)
    weights = (sch[mask].astype(np.float32) + vch[mask].astype(np.float32)) * 0.5
    for hv, ww in zip(hs, weights):
        hist[hv] += ww

    p1 = int(np.argmax(hist))
    s1 = float(hist[p1])
    lo = max(0, p1 - 16)
    hi = min(180, p1 + 17)
    hist2 = hist.copy()
    hist2[lo:hi] = 0
    p2 = int(np.argmax(hist2))
    s2 = float(hist2[p2])

    centers = [float(p1)]
    if s2 >= 0.28 * s1:
        centers.append(float(p2))
    centers.sort()
    return centers[:2]


def stabilize_binary_mask(mask_u8: np.ndarray, max_gap_px: int = 2) -> np.ndarray:
    """Bridge thin dark seams (up to ~2 px) and drop tiny isolated noise."""
    if mask_u8.size == 0:
        return mask_u8
    out = (mask_u8 > 0).astype(np.uint8)
    k = 2 * max(1, int(max_gap_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    if n <= 1:
        return out
    cleaned = np.zeros_like(out)
    min_area = max(12, (max_gap_px + 1) * (max_gap_px + 1))
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == i] = 1
    return cleaned


def trim_binary_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    rs = np.where(mat.sum(axis=1) > 0)[0]
    cs = np.where(mat.sum(axis=0) > 0)[0]
    if len(rs) == 0 or len(cs) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return mat[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1]


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    if union <= 0:
        return 0.0
    return inter / union


def reduce_oversampled_grid(mat: np.ndarray) -> np.ndarray:
    out = mat.copy().astype(np.uint8)
    while True:
        changed = False
        r, c = out.shape

        if r % 2 == 0 and c % 2 == 0 and r >= 2 and c >= 2:
            down = (out.reshape(r // 2, 2, c // 2, 2).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
            up = np.repeat(np.repeat(down, 2, axis=0), 2, axis=1)
            if binary_iou(up, out) >= 0.90 and int(down.sum()) >= 2:
                out = trim_binary_matrix(down)
                changed = True
                continue

        r, c = out.shape
        if r % 2 == 0 and ((r >= 4 and c >= 2) or (r == 2 and c >= 5)):
            down_r = (out.reshape(r // 2, 2, c).mean(axis=1) >= 0.5).astype(np.uint8)
            up_r = np.repeat(down_r, 2, axis=0)
            if binary_iou(up_r, out) >= 0.92 and int(down_r.sum()) >= 2:
                out = trim_binary_matrix(down_r)
                changed = True
                continue

        r, c = out.shape
        if c % 2 == 0 and ((c >= 4 and r >= 2) or (c == 2 and r >= 5)):
            down_c = (out.reshape(r, c // 2, 2).mean(axis=2) >= 0.5).astype(np.uint8)
            up_c = np.repeat(down_c, 2, axis=1)
            if binary_iou(up_c, out) >= 0.92 and int(down_c.sum()) >= 2:
                out = trim_binary_matrix(down_c)
                changed = True
                continue

        if not changed:
            break
    return out


def fit_axis_lines(comp_mask_u8: np.ndarray) -> List[LineSeg]:
    h, w = comp_mask_u8.shape[:2]
    edges = cv2.Canny((comp_mask_u8 * 255).astype(np.uint8), 40, 120)
    min_side = float(min(h, w))
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(8, int(round(min_side * 0.18))),
        minLineLength=max(MIN_FIT_SEG_LEN, int(round(min_side * 0.28))),
        maxLineGap=2,
    )
    if lines is None:
        return []

    axis_lines: List[LineSeg] = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = float(abs(x2 - x1))
        dy = float(abs(y2 - y1))
        # Only keep near axis-aligned edges.
        if dy <= 2.0 and dx >= MIN_FIT_SEG_LEN:
            axis_lines.append((int(x1), int(y1), int(x2), int(y2)))
        elif dx <= 2.0 and dy >= MIN_FIT_SEG_LEN:
            axis_lines.append((int(x1), int(y1), int(x2), int(y2)))
    return axis_lines


def simplify_polyline(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(points) <= 2:
        return points
    out: List[Tuple[int, int]] = [points[0]]
    for i in range(1, len(points) - 1):
        x0, y0 = out[-1]
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if (x0 == x1 == x2) or (y0 == y1 == y2):
            continue
        out.append((x1, y1))
    out.append(points[-1])
    return out


def build_rectilinear_lines_from_poly(poly: np.ndarray) -> List[LineSeg]:
    pts = [(int(p[0]), int(p[1])) for p in poly]
    if len(pts) < 3:
        return []

    orth: List[Tuple[int, int]] = [pts[0]]
    for i in range(1, len(pts)):
        tx, ty = pts[i]
        cx, cy = orth[-1]
        dx = tx - cx
        dy = ty - cy
        if abs(dx) >= abs(dy):
            nx, ny = tx, cy
        else:
            nx, ny = cx, ty
        if (nx, ny) != (cx, cy):
            orth.append((nx, ny))

    # Close polygon with orthogonal corner if needed.
    sx, sy = orth[0]
    lx, ly = orth[-1]
    if lx != sx and ly != sy:
        if abs(lx - sx) < abs(ly - sy):
            orth.append((sx, ly))
        else:
            orth.append((lx, sy))
    if orth[-1] != orth[0]:
        orth.append(orth[0])

    orth = simplify_polyline(orth)
    lines: List[LineSeg] = []
    for i in range(len(orth) - 1):
        x1, y1 = orth[i]
        x2, y2 = orth[i + 1]
        if x1 == x2 or y1 == y2:
            if max(abs(x2 - x1), abs(y2 - y1)) >= MIN_FIT_SEG_LEN:
                lines.append((x1, y1, x2, y2))
        else:
            # Split non-orth segment into an L corner.
            mx, my = x2, y1
            if max(abs(mx - x1), abs(my - y1)) >= MIN_FIT_SEG_LEN:
                lines.append((x1, y1, mx, my))
            if max(abs(x2 - mx), abs(y2 - my)) >= MIN_FIT_SEG_LEN:
                lines.append((mx, my, x2, y2))
    return lines


def extract_closed_contour(comp_mask_u8: np.ndarray, pad: int = 2) -> Optional[np.ndarray]:
    if comp_mask_u8.size == 0:
        return None
    h, w = comp_mask_u8.shape[:2]
    if h < 2 or w < 2:
        return None

    expanded = np.pad(comp_mask_u8.astype(np.uint8), ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    expanded = stabilize_binary_mask(expanded, max_gap_px=2)
    contours, _ = cv2.findContours((expanded * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 8.0:
        return None
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, 0.012 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        x, y, ww, hh = cv2.boundingRect(cnt)
        approx = np.array(
            [[x, y], [x + ww - 1, y], [x + ww - 1, y + hh - 1], [x, y + hh - 1]],
            dtype=np.int32,
        )

    shifted = approx.copy().astype(np.int32)
    shifted[:, 0] = np.clip(shifted[:, 0] - pad, 0, w - 1)
    shifted[:, 1] = np.clip(shifted[:, 1] - pad, 0, h - 1)
    return shifted


def fit_block_outline_lines(comp_mask_u8: np.ndarray, pad: int = 2) -> List[LineSeg]:
    contour = extract_closed_contour(comp_mask_u8, pad=pad)
    if contour is None:
        return fit_axis_lines(comp_mask_u8)

    lines = build_rectilinear_lines_from_poly(contour)
    if not lines:
        return fit_axis_lines(comp_mask_u8)

    shifted: List[LineSeg] = []
    for x1, y1, x2, y2 in lines:
        if max(abs(x2 - x1), abs(y2 - y1)) >= MIN_FIT_SEG_LEN:
            shifted.append((x1, y1, x2, y2))
    return shifted


def axis_edge_lengths(comp_mask_u8: np.ndarray) -> List[float]:
    lines = fit_block_outline_lines(comp_mask_u8, pad=2)
    lengths: List[float] = []
    for x1, y1, x2, y2 in lines:
        dx = float(abs(x2 - x1))
        dy = float(abs(y2 - y1))
        lengths.append(max(dx, dy))
    return lengths


def infer_unit_candidates(comp_mask_u8: np.ndarray, bw: int, bh: int) -> List[float]:
    lengths = axis_edge_lengths(comp_mask_u8)
    cand: List[float] = []
    if lengths:
        arr = np.array(lengths, dtype=np.float32)
        arr = arr[arr >= float(MIN_FIT_SEG_LEN)]
        if arr.size > 0:
            arr.sort()
            for q in (0.20, 0.30, 0.40):
                cand.append(float(np.quantile(arr, q)))
            low_cluster = arr[arr <= float(np.quantile(arr, 0.45)) * 1.15]
            if low_cluster.size > 0:
                cand.append(float(np.median(low_cluster)))

    for k in range(2, 7):
        cand.append(float(bw) / float(k))
        cand.append(float(bh) / float(k))

    # Keep candidates in sane range.
    out: List[float] = []
    for u in cand:
        if u < float(MIN_FIT_SEG_LEN) or u > float(max(bw, bh)):
            continue
        # dedup by proximity.
        if any(abs(u - v) < 1.1 for v in out):
            continue
        out.append(float(u))
    if not out:
        out = [float(max(float(MIN_FIT_SEG_LEN), min(bw, bh) * 0.5))]
    out.sort()
    return out


def infer_unit_len(comp_mask_u8: np.ndarray, bw: int, bh: int) -> float:
    return infer_unit_candidates(comp_mask_u8, bw, bh)[0]


def cells_connected(mat: np.ndarray) -> bool:
    ys, xs = np.where(mat > 0)
    if len(xs) == 0:
        return False
    h, w = mat.shape[:2]
    q: List[Tuple[int, int]] = [(int(ys[0]), int(xs[0]))]
    seen = np.zeros_like(mat, dtype=np.uint8)
    seen[ys[0], xs[0]] = 1
    head = 0
    while head < len(q):
        y, x = q[head]
        head += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if mat[ny, nx] <= 0 or seen[ny, nx] != 0:
                continue
            seen[ny, nx] = 1
            q.append((ny, nx))
    return int(seen.sum()) == int((mat > 0).sum())


def project_occ_to_comp(occ: np.ndarray, bh: int, bw: int) -> np.ndarray:
    nrows, ncols = occ.shape[:2]
    out = np.zeros((bh, bw), dtype=np.uint8)
    ch = bh / nrows
    cw = bw / ncols
    for r in range(nrows):
        yy1 = int(round(r * ch))
        yy2 = int(round((r + 1) * ch))
        for c in range(ncols):
            if int(occ[r, c]) == 0:
                continue
            xx1 = int(round(c * cw))
            xx2 = int(round((c + 1) * cw))
            out[yy1:yy2, xx1:xx2] = 1
    return out


def score_grid_candidate(comp: np.ndarray, occ: np.ndarray) -> float:
    bh, bw = comp.shape[:2]
    proj = project_occ_to_comp(occ, bh, bw)
    iou = binary_iou(comp, proj)
    exp_cells = float(comp.mean()) * float(occ.size)
    area_dev = abs(float(occ.sum()) - exp_cells) / max(1.0, float(occ.size))
    conn_penalty = 0.0 if cells_connected(occ) else 0.18
    score = iou - 0.40 * area_dev - conn_penalty
    return float(score)


def build_occ_for_grid(comp: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    bh, bw = comp.shape[:2]
    occ = np.zeros((nrows, ncols), dtype=np.uint8)
    ch = bh / nrows
    cw = bw / ncols
    for r in range(nrows):
        yy1 = int(round(r * ch))
        yy2 = int(round((r + 1) * ch))
        for c in range(ncols):
            xx1 = int(round(c * cw))
            xx2 = int(round((c + 1) * cw))
            cell_h = max(0, yy2 - yy1)
            cell_w = max(0, xx2 - xx1)
            pad_y = int(round(cell_h * 0.14))
            pad_x = int(round(cell_w * 0.14))
            iy1 = yy1 + pad_y
            iy2 = yy2 - pad_y
            ix1 = xx1 + pad_x
            ix2 = xx2 - pad_x
            if iy2 <= iy1 or ix2 <= ix1:
                iy1, iy2, ix1, ix2 = yy1, yy2, xx1, xx2
            cell = comp[iy1:iy2, ix1:ix2]
            if cell.size == 0:
                continue
            if float(cell.mean()) > 0.34:
                occ[r, c] = 1
    return occ


def infer_shape_from_component(comp_mask_bool: np.ndarray) -> Tuple[List[List[int]], float, Dict[str, object]]:
    ys, xs = np.where(comp_mask_bool)
    if len(xs) == 0:
        return [[1, 1], [1, 1]], 1.0, {
            "unit_candidates": [],
            "grid_candidates": [],
            "best_score": 0.0,
            "chosen_grid": [2, 2],
        }

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    comp = comp_mask_bool[y1:y2, x1:x2].astype(np.uint8)
    comp = stabilize_binary_mask(comp, max_gap_px=1)
    bh, bw = comp.shape[:2]
    unit_candidates = infer_unit_candidates(comp, bw, bh)
    grid_candidates: List[Dict[str, object]] = []
    best_score = -1e9
    best_occ = np.zeros((0, 0), dtype=np.uint8)
    best_unit = unit_candidates[0]
    score_tie_eps = 0.030

    def cell_aspect_error(rows: int, cols: int) -> float:
        cell_h = float(bh) / float(max(1, rows))
        cell_w = float(bw) / float(max(1, cols))
        ratio = cell_w / max(1e-6, cell_h)
        return abs(math.log(max(ratio, 1e-6)))

    for unit in unit_candidates:
        # Keep sane grid search range.
        ncols = max(1, min(8, int(round(bw / max(1.0, unit)))))
        nrows = max(1, min(8, int(round(bh / max(1.0, unit)))))
        occ = build_occ_for_grid(comp, nrows, ncols)
        occ = trim_binary_matrix(occ)
        occ = reduce_oversampled_grid(occ)
        if occ.size == 0:
            continue
        score = score_grid_candidate(comp, occ)
        gc = {
            "unit": float(unit),
            "grid": [int(occ.shape[0]), int(occ.shape[1])],
            "score": float(score),
            "cells": int(occ.sum()),
        }
        grid_candidates.append(gc)
        choose = False
        if score > best_score + 1e-6:
            choose = True
        elif abs(score - best_score) <= score_tie_eps and best_occ.size > 0:
            cur_err = cell_aspect_error(int(occ.shape[0]), int(occ.shape[1]))
            best_err = cell_aspect_error(int(best_occ.shape[0]), int(best_occ.shape[1]))
            if cur_err + 0.012 < best_err:
                choose = True
            elif best_err + 0.012 < cur_err:
                choose = False
            else:
                cur_complex = int(occ.shape[0] * occ.shape[1])
                best_complex = int(best_occ.shape[0] * best_occ.shape[1])
                if cur_complex < best_complex:
                    choose = True
                elif cur_complex == best_complex:
                    cur_cells = int(occ.sum())
                    best_cells = int(best_occ.sum())
                    if cur_cells < best_cells:
                        choose = True
                    elif cur_cells == best_cells and unit > best_unit:
                        choose = True
        elif best_occ.size == 0:
            choose = True

        if choose:
            best_score = score
            best_occ = occ
            best_unit = float(unit)

    # Suspicious downsample protection: large bbox but only 3x3 inferred.
    if best_occ.size > 0 and min(bw, bh) >= 64 and best_occ.shape == (3, 3):
        occ4 = build_occ_for_grid(comp, 4, 4)
        occ4 = trim_binary_matrix(occ4)
        occ4 = reduce_oversampled_grid(occ4)
        if occ4.size > 0:
            score4 = score_grid_candidate(comp, occ4)
            grid_candidates.append(
                {
                    "unit": float(min(bw, bh) / 4.0),
                    "grid": [int(occ4.shape[0]), int(occ4.shape[1])],
                    "score": float(score4),
                    "cells": int(occ4.sum()),
                }
            )
            if score4 > best_score + 0.04:
                best_score = score4
                best_occ = occ4
                best_unit = float(min(bw, bh) / 4.0)

    occ = best_occ
    if occ.size == 0:
        dbg = {
            "unit_candidates": [float(u) for u in unit_candidates],
            "grid_candidates": grid_candidates,
            "best_score": 0.0,
            "chosen_grid": [2, 2],
        }
        return [[1, 1], [1, 1]], best_unit, dbg

    # Game constraint: no mono-cell block.
    if int(occ.sum()) <= 1:
        dbg = {
            "unit_candidates": [float(u) for u in unit_candidates],
            "grid_candidates": sorted(grid_candidates, key=lambda x: float(x["score"]), reverse=True),
            "best_score": float(best_score),
            "chosen_grid": [int(occ.shape[0]), int(occ.shape[1])],
        }
        return [[1, 1], [1, 1]], best_unit, dbg

    # If shape is full square (2x2 vs 3x3 ambiguity), prefer 2x2.
    if occ.shape[0] == occ.shape[1] and int(occ.sum()) == int(occ.shape[0] * occ.shape[1]):
        if best_score < 0.90:
            dbg = {
                "unit_candidates": [float(u) for u in unit_candidates],
                "grid_candidates": sorted(grid_candidates, key=lambda x: float(x["score"]), reverse=True),
                "best_score": float(best_score),
                "chosen_grid": [2, 2],
            }
            return [[1, 1], [1, 1]], best_unit, dbg

    dbg = {
        "unit_candidates": [float(u) for u in unit_candidates],
        "grid_candidates": sorted(grid_candidates, key=lambda x: float(x["score"]), reverse=True),
        "best_score": float(best_score),
        "chosen_grid": [int(occ.shape[0]), int(occ.shape[1])],
    }
    return occ.astype(int).tolist(), best_unit, dbg


def detect_color_blobs(
    roi_bgr: np.ndarray,
    roi_offset: Tuple[int, int],
    cell_size: float,
    color_centers: List[float],
    board_y_range: Tuple[int, int],
) -> Tuple[List[ColorBlob], List[LineSeg], List[Contour], Dict[str, Dict[str, object]]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hch = hsv[:, :, 0]
    sch = hsv[:, :, 1]
    vch = hsv[:, :, 2]
    base_color = (sch > 55) & (vch > 35)
    min_side = 12
    max_side = 90
    min_bbox_area = 900
    left_guard = int(round(cell_size * 0.22))
    hue_gate = 16.0 if len(color_centers) == 1 else 14.0
    min_valid_pixels = 220

    ox, oy = roi_offset
    y_low, y_high = board_y_range
    blobs: List[ColorBlob] = []
    fitted_lines_img: List[LineSeg] = []
    closed_contours_img: List[Contour] = []
    shape_debug_by_blob: Dict[str, Dict[str, object]] = {}
    blob_idx = 1

    for color_idx, center in enumerate(color_centers):
        hue_diff = np.abs(hch.astype(np.int16) - int(round(center)))
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        color_mask = (base_color & (hue_diff <= hue_gate)).astype(np.uint8)
        color_mask = stabilize_binary_mask(color_mask, max_gap_px=2)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < min_side or h < min_side or w > max_side or h > max_side:
                continue
            if (w * h) < min_bbox_area:
                continue
            if x <= left_guard:
                continue

            comp = labels[y : y + h, x : x + w] == i
            sub_h = hch[y : y + h, x : x + w]
            sub_s = sch[y : y + h, x : x + w]
            sub_v = vch[y : y + h, x : x + w]
            valid = comp & (sub_s > 35) & (sub_v > 25)
            if int(valid.sum()) < min_valid_pixels:
                continue

            hvals = sub_h[valid]
            svals = sub_s[valid]
            hue_med = float(np.median(hvals))
            sat_mean = float(np.mean(svals))

            if hue_dist(hue_med, center) > hue_gate:
                continue

            cy = int(y + h * 0.5) + oy
            if cy < y_low or cy > y_high:
                continue

            # Keep fitted rectilinear outline lines for debug visualization.
            local_lines = fit_block_outline_lines(comp.astype(np.uint8), pad=2)
            for lx1, ly1, lx2, ly2 in local_lines:
                fitted_lines_img.append((int(x + ox + lx1), int(y + oy + ly1), int(x + ox + lx2), int(y + oy + ly2)))
            local_contour = extract_closed_contour(comp.astype(np.uint8), pad=2)
            if local_contour is not None:
                contour_points: Contour = []
                for px, py in local_contour.tolist():
                    contour_points.append((int(x + ox + px), int(y + oy + py)))
                if len(contour_points) >= 3:
                    closed_contours_img.append(contour_points)

            shape, unit_len, shape_debug = infer_shape_from_component(comp)
            cur_blob_id = f"blob_{blob_idx:02d}"
            shape_debug_by_blob[cur_blob_id] = shape_debug
            blobs.append(
                ColorBlob(
                    blob_id=cur_blob_id,
                    color_idx=color_idx,
                    area=int(area),
                    hue_median=hue_med,
                    sat_mean=sat_mean,
                    bbox_image=(int(x + ox), int(y + oy), int(w), int(h)),
                    shape=shape,
                    unit_len=unit_len,
                )
            )
            blob_idx += 1

    blobs.sort(key=lambda b: (b.bbox_image[1], b.bbox_image[0]))
    remapped_debug: Dict[str, Dict[str, object]] = {}
    for i, b in enumerate(blobs, start=1):
        old_id = b.blob_id
        b.blob_id = f"blob_{i:02d}"
        if old_id in shape_debug_by_blob:
            remapped_debug[b.blob_id] = shape_debug_by_blob[old_id]
    return blobs, fitted_lines_img, closed_contours_img, remapped_debug


def detect_blocks(
    img_bgr: np.ndarray,
    templates_dir: str,
    board: Optional[BoardDetectionResult] = None,
    energy: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    if board is None:
        board = detect_board_cells(img_bgr, templates_dir)
    corners = board.corners
    cell_size = infer_board_cell_size(corners, board.rows, board.cols)
    img_height = img_bgr.shape[0]
    y_low = 126
    y_high = img_height - 52
    board_y_range = (y_low, y_high)

    x1, y1, x2, y2 = right_side_roi_from_board(img_bgr.shape, corners, cell_size)
    roi_bgr = img_bgr[y1:y2, x1:x2]

    centers = infer_color_centers_from_bars(img_bgr, templates_dir, energy=energy, board=board)
    color_source = "energy_bars"
    if centers is None:
        centers = infer_color_centers_from_roi(roi_bgr)
        color_source = "right_roi_fallback"

    blobs, fitted_lines, closed_contours, shape_debug_by_blob = detect_color_blobs(
        roi_bgr=roi_bgr,
        roi_offset=(x1, y1),
        cell_size=cell_size,
        color_centers=centers,
        board_y_range=board_y_range,
    )

    out = {
        "board_size": [board.rows, board.cols],
        "board_corners": {
            "tl": list(corners["tl"]),
            "tr": list(corners["tr"]),
            "br": list(corners["br"]),
            "bl": list(corners["bl"]),
        },
        "right_side_roi": [x1, y1, x2, y2],
        "board_y_range_filter": [board_y_range[0], board_y_range[1]],
        "color_source": color_source,
        "color_count": len(centers),
        "color_hue_centers": centers,
        "detected_blob_count": len(blobs),
        "blobs": [
            {
                "blob_id": b.blob_id,
                "colour": f"colour_{b.color_idx + 1}",
                "bbox_image": list(b.bbox_image),
                "area": b.area,
                "hue_median": b.hue_median,
                "sat_mean": b.sat_mean,
                "shape": b.shape,
                "unit_len": b.unit_len,
                "shape_debug": shape_debug_by_blob.get(b.blob_id, {}),
            }
            for b in blobs
        ],
    }
    debug = {
        "roi_bgr": roi_bgr,
        "roi_mask": ((cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 1] > 42) & (cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 2] > 28)).astype(np.uint8),
        "fitted_lines": fitted_lines,
        "closed_contours": closed_contours,
        "shape_debug_by_blob": shape_debug_by_blob,
    }
    return out, debug


def save_debug_images(
    img_bgr: np.ndarray,
    result: Dict[str, object],
    debug: Dict[str, object],
    out_prefix: str,
) -> None:
    dbg = img_bgr.copy()
    x1, y1, x2, y2 = result["right_side_roi"]
    cv2.rectangle(dbg, (x1, y1), (x2 - 1, y2 - 1), (180, 180, 180), 1)

    palette = [(0, 220, 0), (0, 140, 255)]
    for b in result["blobs"]:
        x, y, w, h = b["bbox_image"]
        ci = int(b["colour"].split("_")[1]) - 1
        color = palette[ci % len(palette)]
        cv2.rectangle(dbg, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(
            dbg,
            f"{b['blob_id']} {b['colour']}",
            (x, max(14, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(f"{out_prefix}_blocks_debug.png", dbg)
    cv2.imwrite(f"{out_prefix}_right_roi.png", debug["roi_bgr"])
    cv2.imwrite(f"{out_prefix}_right_roi_mask.png", (debug["roi_mask"] * 255).astype(np.uint8))

    # Line fitting debug overlay.
    line_dbg = img_bgr.copy()
    for x1, y1, x2, y2 in debug["fitted_lines"]:
        is_h = abs(y2 - y1) <= abs(x2 - x1)
        col = (0, 255, 255) if is_h else (255, 220, 0)
        cv2.line(line_dbg, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    cv2.imwrite(f"{out_prefix}_linefit_debug.png", line_dbg)

    # Closed contour debug overlay.
    contour_dbg = img_bgr.copy()
    for contour in debug["closed_contours"]:
        pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(contour_dbg, [pts], True, (255, 80, 180), 2, cv2.LINE_AA)
    cv2.imwrite(f"{out_prefix}_contour_debug.png", contour_dbg)

    # Grid-candidate debug overlay.
    grid_dbg = img_bgr.copy()
    for b in result["blobs"]:
        x, y, w, h = b["bbox_image"]
        sdbg = b.get("shape_debug", {})
        chosen = sdbg.get("chosen_grid", [0, 0])
        best_score = float(sdbg.get("best_score", 0.0))
        cv2.rectangle(grid_dbg, (x, y), (x + w - 1, y + h - 1), (80, 240, 255), 1)
        label = f"{b['blob_id']} g={chosen[0]}x{chosen[1]} s={best_score:.3f}"
        cv2.putText(
            grid_dbg,
            label,
            (x, max(14, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (80, 240, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{out_prefix}_grid_debug.png", grid_dbg)


def default_json_path(input_path: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("outputs", f"{base}_blocks.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect colored block blobs on board-right area (without block_UI or shape recognition)."
    )
    parser.add_argument("input", help="Path to screenshot image")
    parser.add_argument("--templates", default="templates", help="Template directory")
    parser.add_argument("--json-out", default=None, help="Output JSON path")
    parser.add_argument("--save-debug", action="store_true", help="Save debug images")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {args.input}")

    result, debug = detect_blocks(img, args.templates)

    out_json = args.json_out or default_json_path(args.input)
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved blocks JSON: {out_json}")
    print(
        f"Detected blobs: {result['detected_blob_count']} "
        f"(colors={result['color_count']} source={result['color_source']})"
    )

    if args.save_debug:
        base = os.path.splitext(out_json)[0]
        save_debug_images(img, result, debug, base)
        print(f"Saved debug image: {base}_blocks_debug.png")
        print(f"Saved line-fit debug: {base}_linefit_debug.png")
        print(f"Saved contour debug: {base}_contour_debug.png")
        print(f"Saved grid debug: {base}_grid_debug.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
