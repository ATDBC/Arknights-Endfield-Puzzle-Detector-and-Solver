import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from detect_board_tags import TagMatch, detect_tags_on_canonical, resize_to_canonical


Point = Tuple[int, int]
LabelGrid = List[List[str]]


GRAY_PRIOR_S_MAX = 20.0
GRAY_PRIOR_V_MIN = 45.0
GRAY_PRIOR_G_MIN = 40.0

LOCK_PRIOR_S_MIN = 60.0
LOCK_PRIOR_V_MIN = 90.0
LOCK_PRIOR_G_MIN = 70.0

GRAY_FALLBACK_S_MAX = 30.0
GRAY_FALLBACK_V_MIN = 35.0
GRAY_FALLBACK_G_MIN = 32.0
GRAY_SCORE_MIN = 0.10
GRAY_MARGIN_MIN = 0.03

UNCERTAIN_LOWCONF_MAX = 0.08


@dataclass
class BoardDetectionResult:
    rows: int
    cols: int
    corners: Dict[str, Point]
    warped_board: np.ndarray
    cell_types: LabelGrid
    gray_matrix: List[List[int]]
    lock_matrix: List[List[int]]


def map_corners_to_original(corners: Dict[str, Point], sx: float, sy: float) -> Dict[str, Point]:
    return {
        "tl": (int(round(corners["tl"][0] * sx)), int(round(corners["tl"][1] * sy))),
        "tr": (int(round(corners["tr"][0] * sx)), int(round(corners["tr"][1] * sy))),
        "br": (int(round(corners["br"][0] * sx)), int(round(corners["br"][1] * sy))),
        "bl": (int(round(corners["bl"][0] * sx)), int(round(corners["bl"][1] * sy))),
    }


def load_board_templates(templates_dir: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in ("empty", "gray", "lock"):
        path = os.path.join(templates_dir, f"board_{name}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Template not found: {path}")
        out[name] = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return out


def template_reference_cell_size(templates_gray: Dict[str, np.ndarray]) -> float:
    sizes: List[float] = []
    for t in templates_gray.values():
        th, tw = t.shape[:2]
        sizes.append(0.5 * (tw + th))
    return float(np.mean(sizes))


def infer_grid_shape_from_corners(
    corners: Dict[str, Point],
    template_cell_size: float,
) -> Tuple[int, int]:
    tl = np.array(corners["tl"], dtype=np.float32)
    tr = np.array(corners["tr"], dtype=np.float32)
    br = np.array(corners["br"], dtype=np.float32)
    bl = np.array(corners["bl"], dtype=np.float32)

    board_w = float(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    board_h = float(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    ref = max(1.0, template_cell_size)

    cols = int(round(board_w / ref))
    rows = int(round(board_h / ref))

    # Keep sane bounds for puzzle boards.
    cols = max(3, min(8, cols))
    rows = max(3, min(8, rows))
    return rows, cols


def board_corners_from_matches(matches: Dict[str, TagMatch]) -> Dict[str, Point]:
    return {
        "tl": matches["left_upper"].board_corner,
        "tr": matches["right_upper"].board_corner,
        "br": matches["right_lower"].board_corner,
        "bl": matches["left_lower"].board_corner,
    }


def warp_board_region(img_bgr: np.ndarray, corners: Dict[str, Point]) -> np.ndarray:
    tl = np.array(corners["tl"], dtype=np.float32)
    tr = np.array(corners["tr"], dtype=np.float32)
    br = np.array(corners["br"], dtype=np.float32)
    bl = np.array(corners["bl"], dtype=np.float32)

    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(width, 16)
    height = max(height, 16)

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, M, (width, height), flags=cv2.INTER_LINEAR)


def inner_cell_crop(cell: np.ndarray, margin_ratio: float = 0.14) -> np.ndarray:
    h, w = cell.shape[:2]
    mx = max(1, int(round(w * margin_ratio)))
    my = max(1, int(round(h * margin_ratio)))
    x1 = min(w - 2, mx)
    y1 = min(h - 2, my)
    x2 = max(x1 + 1, w - mx)
    y2 = max(y1 + 1, h - my)
    return cell[y1:y2, x1:x2]


def inner_cell_crop_with_rect(cell: np.ndarray, margin_ratio: float = 0.14) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = cell.shape[:2]
    mx = max(1, int(round(w * margin_ratio)))
    my = max(1, int(round(h * margin_ratio)))
    x1 = min(w - 2, mx)
    y1 = min(h - 2, my)
    x2 = max(x1 + 1, w - mx)
    y2 = max(y1 + 1, h - my)
    return cell[y1:y2, x1:x2], (x1, y1, x2, y2)


def template_score(cell_gray: np.ndarray, tpl_gray: np.ndarray) -> float:
    resized = cv2.resize(tpl_gray, (cell_gray.shape[1], cell_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return float(cv2.matchTemplate(cell_gray, resized, cv2.TM_CCOEFF_NORMED)[0, 0])


def classify_single_cell(cell_bgr: np.ndarray, templates_gray: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, Any]]:
    cell_hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    cell_gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    s_mean = float(cell_hsv[:, :, 1].mean())
    v_mean = float(cell_hsv[:, :, 2].mean())
    g_mean = float(cell_gray.mean())

    scores = {
        "empty": template_score(cell_gray, templates_gray["empty"]),
        "gray": template_score(cell_gray, templates_gray["gray"]),
        "lock": template_score(cell_gray, templates_gray["lock"]),
    }

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label = str(ordered[0][0])
    top_score = float(ordered[0][1])
    second_label = str(ordered[1][0])
    second_score = float(ordered[1][1])
    margin = top_score - second_score

    debug: Dict[str, Any] = {
        "scores": {k: float(v) for k, v in scores.items()},
        "top_label": top_label,
        "top_score": top_score,
        "second_label": second_label,
        "second_score": second_score,
        "score_margin": float(margin),
        "hsv_mean": {"s": s_mean, "v": v_mean},
        "gray_mean": g_mean,
        "rule_path": "",
    }

    if s_mean > LOCK_PRIOR_S_MIN and v_mean > LOCK_PRIOR_V_MIN and g_mean > LOCK_PRIOR_G_MIN:
        debug["rule_path"] = "lock_strong_prior"
        return "lock", debug

    if s_mean < GRAY_PRIOR_S_MAX and v_mean > GRAY_PRIOR_V_MIN and g_mean > GRAY_PRIOR_G_MIN:
        debug["rule_path"] = "gray_strong_prior"
        return "gray", debug

    if top_label == "gray":
        gray_gate = (
            scores["gray"] >= GRAY_SCORE_MIN
            and margin >= GRAY_MARGIN_MIN
            and s_mean <= GRAY_FALLBACK_S_MAX
            and v_mean >= GRAY_FALLBACK_V_MIN
            and g_mean >= GRAY_FALLBACK_G_MIN
        )
        if gray_gate:
            debug["rule_path"] = "gray_fallback_gate_pass"
            return "gray", debug

        fallback_label = "empty"
        debug["rule_path"] = f"gray rejected by fallback gate -> {fallback_label}"
        return fallback_label, debug

    if top_score < UNCERTAIN_LOWCONF_MAX:
        debug["rule_path"] = "low_confidence_force_empty"
        return "empty", debug

    debug["rule_path"] = "template_fallback_top1"
    return top_label, debug




def classify_grid(
    warped_bgr: np.ndarray,
    rows: int,
    cols: int,
    templates_gray: Dict[str, np.ndarray],
) -> Tuple[LabelGrid, List[List[int]], List[List[int]], List[List[Dict[str, Any]]]]:
    h, w = warped_bgr.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    labels: LabelGrid = []
    gray_matrix: List[List[int]] = []
    lock_matrix: List[List[int]] = []
    cell_debug: List[List[Dict[str, Any]]] = []

    for r in range(rows):
        label_row: List[str] = []
        gray_row: List[int] = []
        lock_row: List[int] = []
        debug_row: List[Dict[str, Any]] = []
        for c in range(cols):
            x1 = int(round(c * cell_w))
            x2 = int(round((c + 1) * cell_w))
            y1 = int(round(r * cell_h))
            y2 = int(round((r + 1) * cell_h))
            raw_cell = warped_bgr[y1:y2, x1:x2]
            cell, crop_rect = inner_cell_crop_with_rect(raw_cell)

            label, cdbg = classify_single_cell(cell, templates_gray)
            cdbg["cell_rc"] = [r + 1, c + 1]
            cdbg["cell_rect_warped"] = [x1, y1, x2, y2]
            cdbg["crop_rect_local"] = [crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]]
            cdbg["crop_size"] = [int(cell.shape[1]), int(cell.shape[0])]
            label_row.append(label)
            gray_row.append(1 if label == "gray" else 0)
            lock_row.append(1 if label == "lock" else 0)
            debug_row.append(cdbg)
        labels.append(label_row)
        gray_matrix.append(gray_row)
        lock_matrix.append(lock_row)
        cell_debug.append(debug_row)

    return labels, gray_matrix, lock_matrix, cell_debug


def annotate_warped_grid(warped_bgr: np.ndarray, labels: LabelGrid) -> np.ndarray:
    out = warped_bgr.copy()
    rows = len(labels)
    cols = len(labels[0]) if rows else 0
    if rows == 0 or cols == 0:
        return out

    h, w = out.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    color_map = {
        "empty": (60, 220, 60),
        "gray": (230, 170, 20),
        "lock": (30, 80, 240),
    }
    short = {"empty": "E", "gray": "G", "lock": "L"}

    for c in range(cols + 1):
        x = int(round(c * cell_w))
        cv2.line(out, (x, 0), (x, h - 1), (230, 230, 230), 1, cv2.LINE_AA)
    for r in range(rows + 1):
        y = int(round(r * cell_h))
        cv2.line(out, (0, y), (w - 1, y), (230, 230, 230), 1, cv2.LINE_AA)

    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w))
            x2 = int(round((c + 1) * cell_w))
            y1 = int(round(r * cell_h))
            y2 = int(round((r + 1) * cell_h))
            label = labels[r][c]
            center = ((x1 + x2) // 2 - 8, (y1 + y2) // 2 + 6)
            cv2.putText(
                out,
                short[label],
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color_map[label],
                2,
                cv2.LINE_AA,
            )

    return out


def annotate_confidence_grid(warped_bgr: np.ndarray, labels: LabelGrid, cell_debug: List[List[Dict[str, Any]]]) -> np.ndarray:
    out = warped_bgr.copy()
    rows = len(labels)
    cols = len(labels[0]) if rows else 0
    if rows == 0 or cols == 0:
        return out

    h, w = out.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    color_map = {
        "empty": (60, 220, 60),
        "gray": (230, 170, 20),
        "lock": (30, 80, 240),
    }
    short = {"empty": "E", "gray": "G", "lock": "L"}

    for c in range(cols + 1):
        x = int(round(c * cell_w))
        cv2.line(out, (x, 0), (x, h - 1), (230, 230, 230), 1, cv2.LINE_AA)
    for r in range(rows + 1):
        y = int(round(r * cell_h))
        cv2.line(out, (0, y), (w - 1, y), (230, 230, 230), 1, cv2.LINE_AA)

    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w))
            x2 = int(round((c + 1) * cell_w))
            y1 = int(round(r * cell_h))
            y2 = int(round((r + 1) * cell_h))
            label = labels[r][c]
            dbg = cell_debug[r][c] if r < len(cell_debug) and c < len(cell_debug[r]) else {}
            top_score = float(dbg.get("top_score", 0.0))
            margin = float(dbg.get("score_margin", 0.0))
            txt = f"{short[label]} {top_score:.2f}/{margin:.2f}"
            cv2.putText(
                out,
                txt,
                (x1 + 4, min(y2 - 6, y1 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color_map[label],
                1,
                cv2.LINE_AA,
            )
    return out


def detect_board_cells(img_bgr: np.ndarray, templates_dir: str) -> BoardDetectionResult:
    canonical_img, sx, sy = resize_to_canonical(img_bgr)
    tag_matches = detect_tags_on_canonical(canonical_img, templates_dir)
    corners_canonical = board_corners_from_matches(tag_matches)
    warped = warp_board_region(canonical_img, corners_canonical)

    templates_gray = load_board_templates(templates_dir)
    ref_cell = template_reference_cell_size(templates_gray)
    rows, cols = infer_grid_shape_from_corners(corners_canonical, ref_cell)
    labels, gray, lock, _ = classify_grid(warped, rows, cols, templates_gray)
    corners_original = map_corners_to_original(corners_canonical, sx, sy)

    return BoardDetectionResult(
        rows=rows,
        cols=cols,
        corners=corners_original,
        warped_board=warped,
        cell_types=labels,
        gray_matrix=gray,
        lock_matrix=lock,
    )


def default_paths(input_image: str) -> Tuple[str, str, str]:
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_image))[0]
    crop_path = os.path.join("outputs", f"{base}_board_crop.png")
    anno_path = os.path.join("outputs", f"{base}_board_cells.png")
    json_path = os.path.join("outputs", f"{base}_board_cells.json")
    return crop_path, anno_path, json_path


def default_debug_paths(json_path: str) -> Tuple[str, str]:
    base, _ = os.path.splitext(json_path)
    return f"{base}_conf_debug.png", f"{base}_debug.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop board from screenshot and classify cells (empty/gray/lock).")
    parser.add_argument("input", help="Path to screenshot image")
    parser.add_argument("--templates", default="templates", help="Path to template directory")
    parser.add_argument("--crop-out", default=None, help="Path to save warped board crop image")
    parser.add_argument("--anno-out", default=None, help="Path to save annotated board image")
    parser.add_argument("--json-out", default=None, help="Path to save detection JSON")
    parser.add_argument("--save-debug", action="store_true", help="Save confidence debug image + JSON")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {args.input}")

    canonical_img, sx, sy = resize_to_canonical(img_bgr)
    tag_matches = detect_tags_on_canonical(canonical_img, args.templates)
    corners_canonical = board_corners_from_matches(tag_matches)
    warped = warp_board_region(canonical_img, corners_canonical)
    templates_gray = load_board_templates(args.templates)
    ref_cell = template_reference_cell_size(templates_gray)
    rows, cols = infer_grid_shape_from_corners(corners_canonical, ref_cell)
    labels, gray, lock, cell_debug = classify_grid(warped, rows, cols, templates_gray)
    corners_original = map_corners_to_original(corners_canonical, sx, sy)

    result = BoardDetectionResult(
        rows=rows,
        cols=cols,
        corners=corners_original,
        warped_board=warped,
        cell_types=labels,
        gray_matrix=gray,
        lock_matrix=lock,
    )
    crop_path_default, anno_path_default, json_path_default = default_paths(args.input)
    crop_path = args.crop_out or crop_path_default
    anno_path = args.anno_out or anno_path_default
    json_path = args.json_out or json_path_default
    conf_debug_path, debug_json_path = default_debug_paths(json_path)

    for p in (crop_path, anno_path, json_path, conf_debug_path, debug_json_path):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    if not cv2.imwrite(crop_path, result.warped_board):
        raise RuntimeError(f"Failed to write cropped board image: {crop_path}")
    annotated = annotate_warped_grid(result.warped_board, result.cell_types)
    if not cv2.imwrite(anno_path, annotated):
        raise RuntimeError(f"Failed to write annotated board image: {anno_path}")

    payload = {
        "rows": result.rows,
        "cols": result.cols,
        "board_corners": {
            "tl": list(result.corners["tl"]),
            "tr": list(result.corners["tr"]),
            "br": list(result.corners["br"]),
            "bl": list(result.corners["bl"]),
        },
        "cell_types": result.cell_types,
        "gray_matrix": result.gray_matrix,
        "lock_matrix": result.lock_matrix,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.save_debug:
        conf_img = annotate_confidence_grid(result.warped_board, result.cell_types, cell_debug)
        if not cv2.imwrite(conf_debug_path, conf_img):
            raise RuntimeError(f"Failed to write confidence debug image: {conf_debug_path}")

        debug_payload = {
            "rows": result.rows,
            "cols": result.cols,
            "cell_debug": cell_debug,
        }
        with open(debug_json_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved board crop: {crop_path}")
    print(f"Saved board annotation: {anno_path}")
    print(f"Saved board JSON: {json_path}")
    if args.save_debug:
        print(f"Saved confidence debug image: {conf_debug_path}")
        print(f"Saved debug JSON: {debug_json_path}")
    print(f"Inferred board size: {result.rows} x {result.cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
