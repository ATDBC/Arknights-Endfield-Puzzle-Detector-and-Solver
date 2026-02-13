import argparse
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
CANONICAL_W = 1709
CANONICAL_H = 961


@dataclass(frozen=True)
class TagSpec:
    name: str
    filename: str
    board_corner_mode: str
    quadrant: str


@dataclass
class TagMatch:
    tag_name: str
    score: float
    rect: Rect
    board_corner: Point
    scale: float


def resize_to_canonical(img_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    h, w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (CANONICAL_W, CANONICAL_H), interpolation=cv2.INTER_LINEAR)
    sx = float(w / CANONICAL_W)
    sy = float(h / CANONICAL_H)
    return resized, sx, sy


def map_rect_to_original(rect: Rect, sx: float, sy: float) -> Rect:
    x, y, w, h = rect
    x0 = int(round(x * sx))
    y0 = int(round(y * sy))
    x1 = int(round((x + w) * sx))
    y1 = int(round((y + h) * sy))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def map_point_to_original(p: Point, sx: float, sy: float) -> Point:
    return (int(round(p[0] * sx)), int(round(p[1] * sy)))


TAG_SPECS: List[TagSpec] = [
    TagSpec(
        name="left_upper",
        filename="tag_left_upper.png",
        board_corner_mode="bottom_right",
        quadrant="left_upper",
    ),
    TagSpec(
        name="left_lower",
        filename="tag_left_lower.png",
        board_corner_mode="top_right",
        quadrant="left_lower",
    ),
    TagSpec(
        name="right_upper",
        filename="tag_right_upper.png",
        board_corner_mode="bottom_left",
        quadrant="right_upper",
    ),
    TagSpec(
        name="right_lower",
        filename="tag_right_lower.png",
        board_corner_mode="top_left",
        quadrant="right_lower",
    ),
]


def auto_canny(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(blur))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if upper <= lower:
        lower, upper = 40, 120
    return cv2.Canny(blur, lower, upper)


def board_corner_from_rect(rect: Rect, mode: str) -> Point:
    x, y, w, h = rect
    if mode == "bottom_right":
        return (x + w - 1, y + h - 1)
    if mode == "top_right":
        return (x + w - 1, y)
    if mode == "bottom_left":
        return (x, y + h - 1)
    if mode == "top_left":
        return (x, y)
    raise ValueError(f"Unknown board corner mode: {mode}")


def centered_window(cx: int, cy: int, radius: int, w: int, h: int) -> Rect:
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    return (x1, y1, x2, y2)


def radius_schedule(w: int, h: int) -> List[int]:
    min_dim = min(w, h)
    max_dim = max(w, h)
    start = max(100, int(min_dim * 0.12))
    end = int(max_dim * 0.52)
    count = 11
    radii = np.linspace(start, end, count, dtype=np.int32).tolist()
    return sorted(set(radii))


def build_scaled_edge_templates(template_gray: np.ndarray, image_h: int) -> List[Tuple[float, np.ndarray, int, int]]:
    base = image_h / 1080.0
    scales = [base * m for m in [0.72, 0.82, 0.92, 1.00, 1.08, 1.18, 1.28]]
    out: List[Tuple[float, np.ndarray, int, int]] = []
    for scale in scales:
        tw = int(round(template_gray.shape[1] * scale))
        th = int(round(template_gray.shape[0] * scale))
        if tw < 14 or th < 14:
            continue
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(template_gray, (tw, th), interpolation=interp)
        out.append((scale, auto_canny(scaled), tw, th))
    return out


def quadrant_ok(point: Point, cx: int, cy: int, quadrant: str, margin: int) -> bool:
    x, y = point
    if quadrant == "left_upper":
        return x <= cx - margin and y <= cy - margin
    if quadrant == "left_lower":
        return x <= cx - margin and y >= cy + margin
    if quadrant == "right_upper":
        return x >= cx + margin and y <= cy - margin
    if quadrant == "right_lower":
        return x >= cx + margin and y >= cy + margin
    return False


def add_candidate_with_nms(
    candidates: List[TagMatch],
    candidate: TagMatch,
    dist_thresh: int,
    max_keep: int,
) -> None:
    cx, cy = candidate.board_corner
    for old in candidates:
        ox, oy = old.board_corner
        if abs(cx - ox) <= dist_thresh and abs(cy - oy) <= dist_thresh:
            if candidate.score > old.score:
                old.score = candidate.score
                old.rect = candidate.rect
                old.board_corner = candidate.board_corner
                old.scale = candidate.scale
            return
    candidates.append(candidate)
    candidates.sort(key=lambda m: m.score, reverse=True)
    if len(candidates) > max_keep:
        del candidates[max_keep:]


def detect_candidates_for_tag(
    edge_img: np.ndarray,
    template_gray: np.ndarray,
    spec: TagSpec,
) -> List[TagMatch]:
    h, w = edge_img.shape
    cx, cy = w // 2, h // 2
    margin = max(18, int(min(w, h) * 0.04))

    radii = radius_schedule(w, h)
    scaled_templates = build_scaled_edge_templates(template_gray, h)
    candidates: List[TagMatch] = []
    score_threshold = 0.56

    for radius in radii:
        x1, y1, x2, y2 = centered_window(cx, cy, radius, w, h)
        roi = edge_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        for scale, tpl_edge, tw, th in scaled_templates:
            if tw >= roi.shape[1] or th >= roi.shape[0]:
                continue
            res = cv2.matchTemplate(roi, tpl_edge, cv2.TM_CCOEFF_NORMED)
            # Extract multiple local peaks so expansion keeps alternatives for non-greedy global choice.
            peak_map = res.copy()
            peak_keep = 3
            for _ in range(peak_keep):
                _, max_val, _, max_loc = cv2.minMaxLoc(peak_map)
                if max_val < score_threshold:
                    break
                rx, ry = int(max_loc[0]), int(max_loc[1])
                rect = (rx + x1, ry + y1, tw, th)
                corner = board_corner_from_rect(rect, spec.board_corner_mode)
                if quadrant_ok(corner, cx, cy, spec.quadrant, margin):
                    add_candidate_with_nms(
                        candidates=candidates,
                        candidate=TagMatch(
                            tag_name=spec.name,
                            score=float(max_val),
                            rect=rect,
                            board_corner=corner,
                            scale=scale,
                        ),
                        dist_thresh=max(16, int(min(tw, th) * 0.35)),
                        max_keep=36,
                    )
                sup = max(8, min(tw, th) // 3)
                cv2.rectangle(
                    peak_map,
                    (max(0, rx - sup), max(0, ry - sup)),
                    (min(peak_map.shape[1] - 1, rx + sup), min(peak_map.shape[0] - 1, ry + sup)),
                    -1,
                    -1,
                )

    if not candidates:
        # Fallback: keep the strongest candidate from full-image search.
        # This keeps the script usable even when threshold/quadrant filters are too strict.
        best: TagMatch | None = None
        for scale, tpl_edge, tw, th in scaled_templates:
            if tw >= w or th >= h:
                continue
            res = cv2.matchTemplate(edge_img, tpl_edge, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            rect = (int(max_loc[0]), int(max_loc[1]), tw, th)
            corner = board_corner_from_rect(rect, spec.board_corner_mode)
            m = TagMatch(
                tag_name=spec.name,
                score=float(max_val),
                rect=rect,
                board_corner=corner,
                scale=scale,
            )
            if best is None or m.score > best.score:
                best = m
        if best is None:
            raise RuntimeError(f"Failed to detect candidates for {spec.name}")
        candidates = [best]

    candidates.sort(key=lambda m: m.score, reverse=True)
    return candidates


def evaluate_combination(
    lu: TagMatch,
    ll: TagMatch,
    ru: TagMatch,
    rl: TagMatch,
    w: int,
    h: int,
) -> float | None:
    tl = lu.board_corner
    bl = ll.board_corner
    tr = ru.board_corner
    br = rl.board_corner

    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        return None

    left_x = 0.5 * (tl[0] + bl[0])
    right_x = 0.5 * (tr[0] + br[0])
    top_y = 0.5 * (tl[1] + tr[1])
    bottom_y = 0.5 * (bl[1] + br[1])
    width = right_x - left_x
    height = bottom_y - top_y
    if width <= 120 or height <= 120:
        return None
    if width > w * 0.85 or height > h * 0.90:
        return None
    ratio = width / max(1.0, height)
    if ratio < 0.45 or ratio > 1.45:
        return None

    cx, cy = w * 0.5, h * 0.5
    if not (left_x <= cx <= right_x and top_y <= cy <= bottom_y):
        return None

    rect_err = abs(tl[0] - bl[0]) + abs(tr[0] - br[0]) + abs(tl[1] - tr[1]) + abs(bl[1] - br[1])
    span_err = abs((tr[0] - tl[0]) - (br[0] - bl[0])) + abs((bl[1] - tl[1]) - (br[1] - tr[1]))
    board_cx = 0.25 * (tl[0] + tr[0] + bl[0] + br[0])
    board_cy = 0.25 * (tl[1] + tr[1] + bl[1] + br[1])
    center_offset = np.hypot(board_cx - cx, board_cy - cy) / max(1.0, min(w, h))

    raw_score = lu.score + ll.score + ru.score + rl.score
    score = raw_score - 0.0045 * rect_err - 0.0035 * span_err - 0.30 * center_offset
    return score


def choose_best_non_greedy(
    pools: Dict[str, List[TagMatch]],
    w: int,
    h: int,
) -> Dict[str, TagMatch]:
    lu_list = pools["left_upper"][:14]
    ll_list = pools["left_lower"][:14]
    ru_list = pools["right_upper"][:14]
    rl_list = pools["right_lower"][:14]

    best_score: float | None = None
    best_combo: Tuple[TagMatch, TagMatch, TagMatch, TagMatch] | None = None

    for lu, ll, ru, rl in itertools.product(lu_list, ll_list, ru_list, rl_list):
        score = evaluate_combination(lu, ll, ru, rl, w, h)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_combo = (lu, ll, ru, rl)

    if best_combo is None:
        # Fallback: strongest per tag.
        return {
            "left_upper": pools["left_upper"][0],
            "left_lower": pools["left_lower"][0],
            "right_upper": pools["right_upper"][0],
            "right_lower": pools["right_lower"][0],
        }

    return {
        "left_upper": best_combo[0],
        "left_lower": best_combo[1],
        "right_upper": best_combo[2],
        "right_lower": best_combo[3],
    }


def detect_tags_on_canonical(img_bgr: np.ndarray, templates_dir: str) -> Dict[str, TagMatch]:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edge_img = auto_canny(img_gray)

    pools: Dict[str, List[TagMatch]] = {}
    for spec in TAG_SPECS:
        template_path = os.path.join(templates_dir, spec.filename)
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            raise FileNotFoundError(f"Template not found: {template_path}")
        template_gray = cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY)
        pools[spec.name] = detect_candidates_for_tag(edge_img, template_gray, spec)

    return choose_best_non_greedy(pools, img_bgr.shape[1], img_bgr.shape[0])


def detect_tags(img_bgr: np.ndarray, templates_dir: str) -> Dict[str, TagMatch]:
    canonical_img, sx, sy = resize_to_canonical(img_bgr)
    matches_canonical = detect_tags_on_canonical(canonical_img, templates_dir)
    mapped: Dict[str, TagMatch] = {}
    for k, m in matches_canonical.items():
        mapped[k] = TagMatch(
            tag_name=m.tag_name,
            score=m.score,
            rect=map_rect_to_original(m.rect, sx, sy),
            board_corner=map_point_to_original(m.board_corner, sx, sy),
            scale=m.scale,
        )
    return mapped


def geometry_warning(matches: Dict[str, TagMatch], img_shape: Tuple[int, int, int]) -> str:
    tl = matches["left_upper"].board_corner
    tr = matches["right_upper"].board_corner
    bl = matches["left_lower"].board_corner
    br = matches["right_lower"].board_corner
    tol = max(14, int(min(img_shape[0], img_shape[1]) * 0.03))

    lx = abs(tl[0] - bl[0])
    rx = abs(tr[0] - br[0])
    ty = abs(tl[1] - tr[1])
    by = abs(bl[1] - br[1])

    if lx <= tol and rx <= tol and ty <= tol and by <= tol:
        return ""
    return (
        f"Warning: tag geometry is not very rectangular "
        f"(left_x_diff={lx}, right_x_diff={rx}, top_y_diff={ty}, bottom_y_diff={by}, tol={tol})."
    )


def annotate_image(img_bgr: np.ndarray, matches: Dict[str, TagMatch]) -> np.ndarray:
    out = img_bgr.copy()
    color_box = (0, 255, 255)
    color_corner = (0, 0, 255)
    color_board = (0, 255, 0)

    label_map = {
        "left_upper": "TL",
        "right_upper": "TR",
        "left_lower": "BL",
        "right_lower": "BR",
    }

    for spec in TAG_SPECS:
        m = matches[spec.name]
        x, y, tw, th = m.rect
        cv2.rectangle(out, (x, y), (x + tw - 1, y + th - 1), color_box, 2)

        txt = f"{spec.name} {m.score:.3f}"
        cv2.putText(
            out,
            txt,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color_box,
            1,
            cv2.LINE_AA,
        )

        cx, cy = m.board_corner
        cv2.circle(out, (cx, cy), 6, color_corner, -1)
        cv2.putText(
            out,
            label_map[spec.name],
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color_corner,
            2,
            cv2.LINE_AA,
        )

    tl = matches["left_upper"].board_corner
    tr = matches["right_upper"].board_corner
    br = matches["right_lower"].board_corner
    bl = matches["left_lower"].board_corner
    cv2.line(out, tl, tr, color_board, 2, cv2.LINE_AA)
    cv2.line(out, tr, br, color_board, 2, cv2.LINE_AA)
    cv2.line(out, br, bl, color_board, 2, cv2.LINE_AA)
    cv2.line(out, bl, tl, color_board, 2, cv2.LINE_AA)
    return out


def default_output_path(input_path: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("outputs", f"{base}_tags.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect board corner tags from puzzle screenshot.")
    parser.add_argument("input", help="Path to screenshot image")
    parser.add_argument(
        "--templates",
        default="templates",
        help="Template directory containing tag_left_upper.png etc.",
    )
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {args.input}")

    matches = detect_tags(img_bgr, args.templates)
    warning_msg = geometry_warning(matches, img_bgr.shape)

    out_img = annotate_image(img_bgr, matches)
    out_path = args.out if args.out else default_output_path(args.input)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ok = cv2.imwrite(out_path, out_img)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {out_path}")

    print(f"Saved annotated image: {out_path}")
    for spec in TAG_SPECS:
        m = matches[spec.name]
        x, y, tw, th = m.rect
        cx, cy = m.board_corner
        print(
            f"{spec.name:>11}: score={m.score:.4f}, rect=(x={x}, y={y}, w={tw}, h={th}), "
            f"board_corner=({cx}, {cy}), scale={m.scale:.3f}"
        )
    if warning_msg:
        print(warning_msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
