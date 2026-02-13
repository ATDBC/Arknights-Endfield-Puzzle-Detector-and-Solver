import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detect_board_cells import BoardDetectionResult, detect_board_cells


Point = Tuple[int, int]


BarRect = Tuple[int, int, int, int, float]


@dataclass
class AnchorDetection:
    axis: int
    count: int
    avg_score: float
    hue: Optional[float]
    color_idx: int = 0
    bars: List[BarRect] = field(default_factory=list)


@dataclass
class GroupDetection:
    anchors: List[AnchorDetection]

    @property
    def total(self) -> int:
        return sum(a.count for a in self.anchors)

    @property
    def confidence(self) -> float:
        if not self.anchors:
            return 0.0
        return float(np.mean([a.avg_score for a in self.anchors]))


def hue_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 180.0 - d)


def load_bar_templates(templates_dir: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    names = ["bar_0_v.png", "bar_1_v.png", "bar_0_h.png", "bar_1_h.png"]
    for n in names:
        p = os.path.join(templates_dir, n)
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Template not found: {p}")
        hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
        # Use S/V channels so shape+intensity are matched while allowing different hues.
        out[n] = hsv[:, :, 1:3]
    return out


def build_expanded_canvas(
    img_bgr: np.ndarray,
    corners: Dict[str, Point],
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int, float, float]]:
    tl = np.array(corners["left_upper"], dtype=np.float32)
    tr = np.array(corners["right_upper"], dtype=np.float32)
    br = np.array(corners["right_lower"], dtype=np.float32)
    bl = np.array(corners["left_lower"], dtype=np.float32)

    board_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    board_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    board_w = max(16, board_w)
    board_h = max(16, board_h)

    cell_w = board_w / cols
    cell_h = board_h / rows
    margin_l = int(round(cell_w * 2.5))
    margin_t = int(round(cell_h * 2.5))
    margin_r = int(round(cell_w * 0.5))
    margin_b = int(round(cell_h * 0.4))

    dst = np.array(
        [
            [margin_l, margin_t],
            [margin_l + board_w - 1, margin_t],
            [margin_l + board_w - 1, margin_t + board_h - 1],
            [margin_l, margin_t + board_h - 1],
        ],
        dtype=np.float32,
    )
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(
        img_bgr,
        M,
        (margin_l + board_w + margin_r, margin_t + board_h + margin_b),
        flags=cv2.INTER_LINEAR,
    )
    return canvas, (margin_l, margin_t, board_w, board_h, cell_w, cell_h)


def extract_top_roi(meta: Tuple[int, int, int, int, float, float], col: int) -> Tuple[int, int, int, int]:
    margin_l, margin_t, _, _, cell_w, cell_h = meta
    # Keep exact grid-cell width to avoid losing bar info near left/right edges.
    x1 = int(round(margin_l + col * cell_w))
    x2 = int(round(margin_l + (col + 1) * cell_w))
    y1 = max(0, int(round(margin_t - cell_h * 2.25)))
    y2 = max(1, int(round(margin_t - cell_h * 0.01)))
    return x1, y1, x2, y2


def extract_left_roi(meta: Tuple[int, int, int, int, float, float], row: int) -> Tuple[int, int, int, int]:
    margin_l, margin_t, _, _, cell_w, cell_h = meta
    x1 = max(0, int(round(margin_l - cell_w * 2.25)))
    x2 = max(1, int(round(margin_l - cell_w * 0.01)))
    # Keep exact grid-cell height to avoid losing bar info near top/bottom edges.
    y1 = int(round(margin_t + row * cell_h))
    y2 = int(round(margin_t + (row + 1) * cell_h))
    return x1, y1, x2, y2


def patch_hue(hsv: np.ndarray, x: int, y: int, tw: int, th: int) -> Optional[float]:
    patch = hsv[y : y + th, x : x + tw]
    if patch.size == 0:
        return None
    mask = (patch[:, :, 1] > 35) & (patch[:, :, 2] > 35)
    if not mask.any():
        return None
    return float(np.median(patch[:, :, 0][mask]))


def border_hue(hsv: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[float]:
    patch = hsv[y : y + h, x : x + w]
    if patch.size == 0:
        return None
    for thickness in (1, 2):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (w - 1, h - 1), 1, thickness=thickness)
        s = patch[:, :, 1]
        v = patch[:, :, 2]
        sel = (mask > 0) & (s > 35) & (v > 35)
        if sel.any():
            return float(np.median(patch[:, :, 0][sel]))
    return None


def build_response_map(
    roi_bgr: np.ndarray,
    tpl_a: np.ndarray,
    tpl_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sv = hsv[:, :, 1:3]
    res_a = cv2.matchTemplate(sv, tpl_a, cv2.TM_CCOEFF_NORMED)
    res_b = cv2.matchTemplate(sv, tpl_b, cv2.TM_CCOEFF_NORMED)
    h = min(res_a.shape[0], res_b.shape[0])
    w = min(res_a.shape[1], res_b.shape[1])
    return np.maximum(res_a[:h, :w], res_b[:h, :w]), hsv


def detect_bar_candidates(roi_bgr: np.ndarray, mode: str) -> List[Dict[str, object]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    vch = hsv[:, :, 2]
    edges_g = cv2.Canny(gray, 60, 160)
    edges_v = cv2.Canny(vch, 60, 160)
    edges = cv2.bitwise_or(edges_g, edges_v)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    bars: List[Dict[str, object]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 4 or h < 4:
            continue

        long_side = max(w, h)
        short_side = min(w, h)
        if long_side < 27 or long_side > 30:
            continue
        if short_side < 7 or short_side > 12:
            continue
        if mode == "top" and w < h:
            continue
        if mode != "top" and h < w:
            continue

        area = float(cv2.contourArea(cnt))
        rect_area = float(w * h)
        if rect_area <= 0.0:
            continue
        fill = area / rect_area
        if fill < 0.35:
            continue

        hue = border_hue(hsv, x, y, w, h)
        bars.append({"x": x, "y": y, "w": w, "h": h, "score": fill, "hue": hue})

    # De-duplicate near identical rectangles.
    bars.sort(key=lambda b: float(b["score"]), reverse=True)
    kept: List[Dict[str, object]] = []
    for b in bars:
        bx, by, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        if any(abs(bx - int(k["x"])) <= 2 and abs(by - int(k["y"])) <= 2 for k in kept):
            continue
        kept.append(b)
    return kept


def cluster_bars_by_hue(bars: List[Dict[str, object]], threshold: float = 20.0) -> List[List[Dict[str, object]]]:
    with_hue = [b for b in bars if b.get("hue") is not None]
    without_hue = [b for b in bars if b.get("hue") is None]

    clusters: List[List[Dict[str, object]]] = []
    for b in sorted(with_hue, key=lambda x: float(x["hue"])):  # type: ignore[index]
        placed = False
        for c in clusters:
            center = float(np.median([float(x["hue"]) for x in c if x.get("hue") is not None]))
            if hue_dist(float(b["hue"]), center) <= threshold:  # type: ignore[index]
                c.append(b)
                placed = True
                break
        if not placed:
            clusters.append([b])

    if without_hue:
        clusters.append(without_hue)
    return clusters


def anchor_from_cluster(
    bars: List[Dict[str, object]],
    mode: str,
    align_tol: int = 3,
) -> Optional[AnchorDetection]:
    if not bars:
        return None

    def pos(v: Dict[str, object]) -> int:
        return int(v["y"]) if mode == "top" else int(v["x"])

    def align(v: Dict[str, object]) -> int:
        return int(v["x"]) if mode == "top" else int(v["y"])

    def expected_next_pos(v: Dict[str, object], gap: int) -> int:
        if mode == "top":
            return int(v["y"]) - int(v["h"]) - gap
        return int(v["x"]) - int(v["w"]) - gap

    bars_sorted = sorted(bars, key=pos, reverse=True)
    best_seq: List[Dict[str, object]] = []
    best_score = -1.0

    for i, start in enumerate(bars_sorted):
        align_ref = align(start)
        subset = [b for b in bars_sorted if abs(align(b) - align_ref) <= align_tol]
        subset = sorted(subset, key=pos, reverse=True)
        if not subset:
            continue

        for s in range(len(subset)):
            seq = [subset[s]]
            cur_idx = s
            while True:
                prev = subset[cur_idx]
                nxt_candidates: List[Tuple[float, int]] = []
                for j in range(cur_idx + 1, len(subset)):
                    cand = subset[j]
                    if not valid_bar_transition(
                        prev_x=int(prev["x"]),
                        prev_y=int(prev["y"]),
                        cur_x=int(cand["x"]),
                        cur_y=int(cand["y"]),
                        mode=mode,
                        bar_w=int(prev["w"]),
                        bar_h=int(prev["h"]),
                        align_tol=align_tol,
                    ):
                        continue
                    exp5 = expected_next_pos(prev, 5)
                    exp6 = expected_next_pos(prev, 6)
                    p = pos(cand)
                    closeness = min(abs(p - exp5), abs(p - exp6))
                    sc = float(cand["score"]) - 0.03 * float(closeness)
                    nxt_candidates.append((sc, j))
                if not nxt_candidates:
                    break
                nxt_candidates.sort(key=lambda t: t[0], reverse=True)
                cur_idx = nxt_candidates[0][1]
                seq.append(subset[cur_idx])

            seq_score = float(len(seq)) + 0.2 * float(np.mean([float(v["score"]) for v in seq]))
            if seq_score > best_score:
                best_score = seq_score
                best_seq = seq

    if not best_seq:
        return None

    hues = [float(b["hue"]) for b in best_seq if b.get("hue") is not None]
    hue = float(np.median(hues)) if hues else None
    scores = [float(b["score"]) for b in best_seq]
    avg_score = float(np.mean(scores)) if scores else -1.0
    axis_val = int(np.median([align(v) for v in best_seq])) if best_seq else 0
    rects: List[BarRect] = [
        (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]), float(b["score"])) for b in best_seq
    ]
    return AnchorDetection(axis=axis_val, count=len(rects), avg_score=avg_score, hue=hue, bars=rects)

def collect_peaks(response: np.ndarray, mode: str, threshold: float = 0.38) -> List[Tuple[float, int, int]]:
    r = response.copy()
    if mode == "top":
        r[: max(0, response.shape[0] - 40), :] = -1.0
    else:
        r[:, : max(0, response.shape[1] - 40)] = -1.0

    peaks: List[Tuple[float, int, int]] = []
    for _ in range(40):
        _, max_val, _, loc = cv2.minMaxLoc(r)
        if max_val < threshold:
            break
        x, y = int(loc[0]), int(loc[1])
        peaks.append((float(max_val), x, y))
        cv2.rectangle(
            r,
            (max(0, x - 4), max(0, y - 4)),
            (min(r.shape[1] - 1, x + 4), min(r.shape[0] - 1, y + 4)),
            -1,
            -1,
        )
    return peaks


def choose_axis_candidates(
    peaks: List[Tuple[float, int, int]],
    mode: str,
    min_sep: int = 12,
    max_candidates: int = 3,
) -> List[int]:
    chosen: List[int] = []
    sorted_peaks = sorted(peaks, key=lambda t: t[0], reverse=True)
    for _, x, y in sorted_peaks:
        axis = x if mode == "top" else y
        if any(abs(axis - c) < min_sep for c in chosen):
            continue
        chosen.append(axis)
        if len(chosen) >= max_candidates:
            break
    return chosen


def valid_bar_transition(
    prev_x: int,
    prev_y: int,
    cur_x: int,
    cur_y: int,
    mode: str,
    bar_w: int,
    bar_h: int,
    align_tol: int = 3,
) -> bool:
    # Enforce strict spacing and alignment constraints:
    # - gap between adjacent bars must be 5 or 6 px
    # - bars must align on the orthogonal axis
    if mode == "top":
        if cur_y >= prev_y:
            return False
        # Use a stable effective thickness for hollow bars.
        # In practice, contour boxes vary by 1-2 px with background, while
        # the playable bar stroke thickness is close to 8 px.
        eff_h = 8
        gap = int(prev_y - (cur_y + eff_h))
        return gap in (5, 6) and abs(cur_x - prev_x) <= align_tol

    if cur_x >= prev_x:
        return False
    eff_w = 8
    gap = int(prev_x - (cur_x + eff_w))
    return gap in (5, 6) and abs(cur_y - prev_y) <= align_tol


def count_anchor(
    response: np.ndarray,
    hsv: np.ndarray,
    axis: int,
    mode: str,
    threshold: float,
) -> AnchorDetection:
    if mode == "top":
        patch_w, patch_h = 27, 10
        axis_len = response.shape[0]
        step_candidates = [15, 16]
    else:
        patch_w, patch_h = 9, 27
        axis_len = response.shape[1]
        step_candidates = [14, 15]

    best_count = 0
    best_avg = -1.0
    best_hue: Optional[float] = None
    best_bars: List[BarRect] = []

    for pitch in step_candidates:
        for offset in range(max(16, pitch + 6)):
            cnt = 0
            scores: List[float] = []
            hues: List[float] = []
            ref_hue: Optional[float] = None
            bars: List[BarRect] = []
            align_ref: Optional[int] = None

            for k in range(10):
                pos = (axis_len - 1) - offset - k * pitch
                if pos < 0:
                    break

                best_local_score = -1.0
                best_x = 0
                best_y = 0
                if mode == "top":
                    for yy in range(max(0, pos - 3), min(response.shape[0], pos + 4)):
                        for xx in range(max(0, axis - 4), min(response.shape[1], axis + 5)):
                            sc = float(response[yy, xx])
                            if sc > best_local_score:
                                best_local_score = sc
                                best_x, best_y = xx, yy
                else:
                    for xx in range(max(0, pos - 3), min(response.shape[1], pos + 4)):
                        for yy in range(max(0, axis - 4), min(response.shape[0], axis + 5)):
                            sc = float(response[yy, xx])
                            if sc > best_local_score:
                                best_local_score = sc
                                best_x, best_y = xx, yy

                if best_local_score < threshold:
                    break

                # Bars in one anchor must be aligned on orthogonal axis.
                align_cur = best_x if mode == "top" else best_y
                if align_ref is None:
                    align_ref = align_cur
                elif abs(align_cur - align_ref) > 3:
                    break

                if bars:
                    px, py, pw, ph, _ = bars[-1]
                    if not valid_bar_transition(
                        prev_x=px,
                        prev_y=py,
                        cur_x=best_x,
                        cur_y=best_y,
                        mode=mode,
                        bar_w=pw,
                        bar_h=ph,
                        align_tol=3,
                    ):
                        break

                hue = patch_hue(hsv, best_x, best_y, patch_w, patch_h)
                if hue is not None:
                    if ref_hue is None:
                        ref_hue = hue
                    elif hue_dist(hue, ref_hue) > 20:
                        break
                    hues.append(hue)

                cnt += 1
                scores.append(best_local_score)
                bars.append((best_x, best_y, patch_w, patch_h, best_local_score))

            avg = float(np.mean(scores)) if scores else -1.0
            hue = float(np.median(hues)) if hues else None
            if cnt > best_count or (cnt == best_count and avg > best_avg):
                best_count, best_avg, best_hue = cnt, avg, hue
                best_bars = bars

    return AnchorDetection(axis=axis, count=best_count, avg_score=best_avg, hue=best_hue, bars=best_bars)


def detect_group_anchors(
    roi_bgr: np.ndarray,
    mode: str,
    templates: Dict[str, np.ndarray],
    threshold: float,
) -> GroupDetection:
    del templates, threshold
    candidates = detect_bar_candidates(roi_bgr, mode)
    clusters = cluster_bars_by_hue(candidates, threshold=20.0)
    anchors: List[AnchorDetection] = []
    for cluster in clusters:
        a = anchor_from_cluster(cluster, mode=mode, align_tol=3)
        if a is not None and a.count > 0:
            anchors.append(a)
    return GroupDetection(anchors=anchors)


def infer_color_centers_from_top(top_groups: List[GroupDetection]) -> List[float]:
    hues: List[float] = []
    weights: List[float] = []
    for g in top_groups:
        for a in g.anchors:
            if a.hue is None or a.count <= 0:
                continue
            hues.append(a.hue)
            weights.append(max(0.0, a.avg_score) * a.count)
    if not hues:
        return [35.0]

    clusters: List[Dict[str, object]] = []
    for h, w in sorted(zip(hues, weights), key=lambda t: t[1], reverse=True):
        placed = False
        for c in clusters:
            center = float(c["center"])
            if hue_dist(h, center) < 20:
                vals = c["vals"]  # type: ignore[assignment]
                vals.append((h, w))
                ww = [vw for _, vw in vals]
                hh = [vh for vh, _ in vals]
                c["sum_w"] = float(c["sum_w"]) + w
                c["center"] = float(np.average(hh, weights=ww))
                placed = True
                break
        if not placed:
            clusters.append({"center": h, "sum_w": w, "vals": [(h, w)]})

    clusters = sorted(clusters, key=lambda c: float(c["sum_w"]), reverse=True)
    if len(clusters) >= 2 and float(clusters[1]["sum_w"]) >= 0.20 * float(clusters[0]["sum_w"]):
        centers = [float(clusters[0]["center"]), float(clusters[1]["center"])]
    else:
        centers = [float(clusters[0]["center"])]
    # Keep stable ordering for colour_1/colour_2.
    centers.sort()
    return centers


def pick_group_anchors(group: GroupDetection, color_centers: List[float]) -> List[AnchorDetection]:
    if not group.anchors:
        return []
    if len(color_centers) == 1:
        return [sorted(group.anchors, key=lambda a: (a.count, a.avg_score), reverse=True)[0]]

    selected: List[AnchorDetection] = []
    used: set[int] = set()
    for center in color_centers:
        best_idx = -1
        best_score = -1e9
        for i, a in enumerate(group.anchors):
            if i in used:
                continue
            d = 0.0 if a.hue is None else hue_dist(a.hue, center)
            score = a.count + 0.8 * a.avg_score - 0.04 * d
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            selected.append(group.anchors[best_idx])
    return selected


def assign_anchor_colors(anchors: List[AnchorDetection], color_centers: List[float]) -> None:
    for a in anchors:
        if len(color_centers) == 1:
            a.color_idx = 0
            continue
        if a.hue is None:
            a.color_idx = 0
            continue
        dists = [hue_dist(a.hue, c) for c in color_centers]
        a.color_idx = int(np.argmin(dists))


def side_confidence(groups: List[List[AnchorDetection]]) -> float:
    vals: List[float] = []
    for g in groups:
        if not g:
            vals.append(0.0)
        else:
            vals.append(float(np.mean([a.avg_score for a in g])))
    return float(np.mean(vals)) if vals else 0.0


def balance_totals(
    cols_sel: List[List[AnchorDetection]],
    rows_sel: List[List[AnchorDetection]],
) -> None:
    sum_cols = sum(sum(a.count for a in g) for g in cols_sel)
    sum_rows = sum(sum(a.count for a in g) for g in rows_sel)
    if sum_cols == sum_rows:
        return

    # Adjust the lower-confidence side toward the higher-confidence side.
    conf_cols = side_confidence(cols_sel)
    conf_rows = side_confidence(rows_sel)

    if conf_cols >= conf_rows:
        target = sum_cols
        side = rows_sel
    else:
        target = sum_rows
        side = cols_sel

    current = sum(sum(a.count for a in g) for g in side)
    diff = current - target
    if diff == 0:
        return

    # Only handle small drift by decrementing weakest anchors.
    # (Typical mismatch is 1-2 due false positives.)
    steps = abs(diff)
    for _ in range(steps):
        candidate = None
        for gi, g in enumerate(side):
            for ai, a in enumerate(g):
                if diff > 0 and a.count <= 0:
                    continue
                score = a.avg_score + 0.1 * a.count
                if candidate is None or score < candidate[0]:
                    candidate = (score, gi, ai)
        if candidate is None:
            break
        _, gi, ai = candidate
        if diff > 0:
            side[gi][ai].count = max(0, side[gi][ai].count - 1)
        else:
            side[gi][ai].count += 1


def groups_to_color_arrays(
    groups: List[List[AnchorDetection]],
    color_count: int,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {i: [] for i in range(color_count)}
    for g in groups:
        vals = [0 for _ in range(color_count)]
        for a in g:
            vals[a.color_idx] += a.count
        for i in range(color_count):
            out[i].append(vals[i])
    return out


def used_bars(anchor: AnchorDetection) -> List[BarRect]:
    if anchor.count <= 0 or not anchor.bars:
        return []
    return anchor.bars[: min(anchor.count, len(anchor.bars))]


def render_energy_bar_debug(
    canvas_bgr: np.ndarray,
    top_rois: List[Tuple[int, int, int, int]],
    left_rois: List[Tuple[int, int, int, int]],
    cols_sel: List[List[AnchorDetection]],
    rows_sel: List[List[AnchorDetection]],
    templates: Dict[str, np.ndarray],
) -> np.ndarray:
    dbg = canvas_bgr.copy()
    # Colors are BGR, keep labels fixed and visible.
    palette = [(0, 230, 0), (0, 150, 255)]
    del templates

    # Draw ROI boxes (1px), helps check extraction windows.
    for (x1, y1, x2, y2) in top_rois:
        cv2.rectangle(dbg, (x1, y1), (x2 - 1, y2 - 1), (160, 160, 160), 1)
    for (x1, y1, x2, y2) in left_rois:
        cv2.rectangle(dbg, (x1, y1), (x2 - 1, y2 - 1), (110, 110, 110), 1)

    # Draw each detected bar location with 1px rectangle.
    for gi, anchors in enumerate(cols_sel):
        x1, y1, _, _ = top_rois[gi]
        for a in anchors:
            color = palette[a.color_idx % len(palette)]
            for bx, by, bw, bh, _ in used_bars(a):
                gx = int(x1 + bx)
                gy = int(y1 + by)
                cv2.rectangle(dbg, (gx, gy), (gx + bw - 1, gy + bh - 1), color, 1)

    for gi, anchors in enumerate(rows_sel):
        x1, y1, _, _ = left_rois[gi]
        for a in anchors:
            color = palette[a.color_idx % len(palette)]
            for bx, by, bw, bh, _ in used_bars(a):
                gx = int(x1 + bx)
                gy = int(y1 + by)
                cv2.rectangle(dbg, (gx, gy), (gx + bw - 1, gy + bh - 1), color, 1)
    return dbg


def detect_energy_bars(
    img_bgr: np.ndarray,
    templates_dir: str,
    return_debug_image: bool = False,
    board: Optional[BoardDetectionResult] = None,
) -> Dict[str, object] | Tuple[Dict[str, object], np.ndarray]:
    if board is None:
        board = detect_board_cells(img_bgr, templates_dir)
    rows, cols = board.rows, board.cols

    corners = {
        "left_upper": board.corners["tl"],
        "right_upper": board.corners["tr"],
        "right_lower": board.corners["br"],
        "left_lower": board.corners["bl"],
    }
    canvas, meta = build_expanded_canvas(img_bgr, corners, rows, cols)
    templates = load_bar_templates(templates_dir)

    # Pass-1 top detection for color-count inference.
    top_groups_lo: List[GroupDetection] = []
    top_rois: List[Tuple[int, int, int, int]] = []
    for c in range(cols):
        x1, y1, x2, y2 = extract_top_roi(meta, c)
        top_rois.append((x1, y1, x2, y2))
        roi = canvas[y1:y2, x1:x2]
        top_groups_lo.append(detect_group_anchors(roi, mode="top", templates=templates, threshold=0.45))

    color_centers = infer_color_centers_from_top(top_groups_lo)
    color_count = len(color_centers)

    # Final pass.
    top_th = 0.45
    left_th = 0.55 if color_count == 1 else 0.50

    top_groups: List[GroupDetection] = []
    for c in range(cols):
        x1, y1, x2, y2 = top_rois[c]
        roi = canvas[y1:y2, x1:x2]
        top_groups.append(detect_group_anchors(roi, mode="top", templates=templates, threshold=top_th))

    left_groups: List[GroupDetection] = []
    left_rois: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        x1, y1, x2, y2 = extract_left_roi(meta, r)
        left_rois.append((x1, y1, x2, y2))
        roi = canvas[y1:y2, x1:x2]
        left_groups.append(detect_group_anchors(roi, mode="left", templates=templates, threshold=left_th))

    cols_sel = [pick_group_anchors(g, color_centers) for g in top_groups]
    rows_sel = [pick_group_anchors(g, color_centers) for g in left_groups]
    for g in cols_sel:
        assign_anchor_colors(g, color_centers)
    for g in rows_sel:
        assign_anchor_colors(g, color_centers)

    balance_totals(cols_sel, rows_sel)

    col_targets = groups_to_color_arrays(cols_sel, color_count)
    row_targets = groups_to_color_arrays(rows_sel, color_count)

    # Convert to standard colour_n keys.
    col_targets_out = {f"colour_{i+1}": col_targets[i] for i in range(color_count)}
    row_targets_out = {f"colour_{i+1}": row_targets[i] for i in range(color_count)}

    col_totals = [sum(col_targets[i][j] for i in range(color_count)) for j in range(cols)]
    row_totals = [sum(row_targets[i][j] for i in range(color_count)) for j in range(rows)]

    result = {
        "rows": rows,
        "cols": cols,
        "top_group_count": cols,
        "left_group_count": rows,
        "color_count": color_count,
        "color_hue_centers": color_centers,
        "column_totals": col_totals,
        "row_totals": row_totals,
        "col_targets": col_targets_out,
        "row_targets": row_targets_out,
        "thresholds": {"top": top_th, "left": left_th},
    }
    if not return_debug_image:
        return result
    debug_img = render_energy_bar_debug(
        canvas_bgr=canvas,
        top_rois=top_rois,
        left_rois=left_rois,
        cols_sel=cols_sel,
        rows_sel=rows_sel,
        templates=templates,
    )
    return result, debug_img


def default_output_path(input_path: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("outputs", f"{base}_energy_bars.json")


def default_debug_path(input_path: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("outputs", f"{base}_energy_bars_debug.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect energy bars above/left of board (RULES step 2/3/4).")
    parser.add_argument("input", help="Path to screenshot image")
    parser.add_argument("--templates", default="templates", help="Path to templates directory")
    parser.add_argument("--json-out", default=None, help="Output JSON path")
    parser.add_argument("--save-debug", action="store_true", help="Save 1px bar-mark debug image")
    parser.add_argument("--debug-out", default=None, help="Debug image path")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {args.input}")

    debug_img: Optional[np.ndarray] = None
    if args.save_debug:
        result, debug_img = detect_energy_bars(img_bgr, args.templates, return_debug_image=True)  # type: ignore[misc]
    else:
        result = detect_energy_bars(img_bgr, args.templates)  # type: ignore[assignment]
    out_path = args.json_out or default_output_path(args.input)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved energy-bar JSON: {out_path}")
    print(f"Detected colors: {result['color_count']}, hue centers={result['color_hue_centers']}")
    print(f"Top groups (columns): {result['column_totals']}")
    print(f"Left groups (rows): {result['row_totals']}")
    if args.save_debug and debug_img is not None:
        dbg_path = args.debug_out or default_debug_path(args.input)
        dbg_dir = os.path.dirname(dbg_path)
        if dbg_dir:
            os.makedirs(dbg_dir, exist_ok=True)
        ok = cv2.imwrite(dbg_path, debug_img)
        if not ok:
            raise RuntimeError(f"Failed to write debug image: {dbg_path}")
        print(f"Saved debug image: {dbg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
