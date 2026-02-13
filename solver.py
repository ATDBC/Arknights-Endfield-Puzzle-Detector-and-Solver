import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple


Coord = Tuple[int, int]


@dataclass
class Block:
    block_id: str
    color: int
    shape: List[List[int]]


@dataclass
class Puzzle:
    rows: int
    cols: int
    row_targets: Dict[int, List[int]]
    col_targets: Dict[int, List[int]]
    gray: List[List[int]]
    locks: Dict[int, List[List[int]]]
    blocks: List[Block]
    color_rgb: Dict[int, Tuple[int, int, int]]


@dataclass
class OrientedBlock:
    block_id: str
    color: int
    cells: List[Coord]
    height: int
    width: int


def _clean_list_literal(text: str):
    # Remove trailing commas before ] to make ast parsing robust.
    text = re.sub(r",\s*\]", "]", text)
    return ast.literal_eval(text)


def _extract_bracket_expr(text: str, start_idx: int) -> Tuple[str, int]:
    if start_idx >= len(text) or text[start_idx] != "[":
        raise ValueError("Expected '[' at start_idx")
    depth = 0
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1], i + 1
    raise ValueError("Unmatched brackets in puzzle text")


def _is_list_of_ints(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, int) for x in v)


def _is_matrix(v) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(_is_list_of_ints(r) for r in v)


def _parse_text_puzzle(raw: str) -> Puzzle:
    # 1) Parse board hints with a tolerant regex because sample files can miss one closing bracket.
    board_entries = []
    board_ranges = []
    board_pat = re.compile(
        r"colour_(\d+)\s*:\s*\[\s*\[([0-9,\s]+)\]\s*,\s*\[([0-9,\s]+)\]\s*\]?\s*(?=(?:,\s*colour_\d+\s*:|}))",
        flags=re.S,
    )
    for m in board_pat.finditer(raw):
        color = int(m.group(1))
        cols_list = [int(x) for x in m.group(2).split(",") if x.strip()]
        rows_list = [int(x) for x in m.group(3).split(",") if x.strip()]
        board_entries.append((m.start(), m.end(), color, [cols_list, rows_list]))
        board_ranges.append((m.start(), m.end()))

    # 2) Parse all well-formed 'colour_n: [...]' entries for locks.
    color_entries = []
    for m in re.finditer(r"colour_(\d+)\s*:\s*", raw):
        # Skip ranges already consumed as board entries.
        if any(a <= m.start() < b for a, b in board_ranges):
            continue
        color = int(m.group(1))
        idx = m.end()
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx >= len(raw) or raw[idx] != "[":
            continue
        try:
            literal, end_idx = _extract_bracket_expr(raw, idx)
            parsed = _clean_list_literal(literal)
            color_entries.append((m.start(), end_idx, color, parsed))
        except ValueError:
            continue

    lock_entries = []
    for start, end, color, parsed in color_entries:
        if _is_matrix(parsed):
            lock_entries.append((start, end, color, parsed))

    if not board_entries:
        raise ValueError("No board hint entries found (colour_n: [[...],[...]]).")

    row_targets: Dict[int, List[int]] = {}
    col_targets: Dict[int, List[int]] = {}
    for _, _, color, parsed in board_entries:
        col_targets[color] = list(parsed[0])
        row_targets[color] = list(parsed[1])

    any_color = next(iter(row_targets.keys()))
    rows = len(row_targets[any_color])
    cols = len(col_targets[any_color])

    for c in row_targets:
        if len(row_targets[c]) != rows or len(col_targets[c]) != cols:
            raise ValueError("Inconsistent board sizes across colors")

    # 2) Parse gray matrix: first full-size matrix not attached to colour_n and before first block.
    first_block_match = re.search(r"colour\s*=\s*\d+", raw)
    first_block_pos = first_block_match.start() if first_block_match else len(raw)

    color_spans = [(s, e) for s, e, _, _ in color_entries]

    def in_color_span(pos: int) -> bool:
        return any(s <= pos < e for s, e in color_spans)

    gray = None
    idx = 0
    while idx < first_block_pos:
        if raw[idx] == "[" and not in_color_span(idx):
            try:
                literal, end_idx = _extract_bracket_expr(raw, idx)
                parsed = _clean_list_literal(literal)
                if (
                    _is_matrix(parsed)
                    and len(parsed) == rows
                    and all(len(r) == cols for r in parsed)
                ):
                    gray = parsed
                    break
                idx = end_idx
                continue
            except Exception:
                pass
        idx += 1

    if gray is None:
        raise ValueError("Gray-cell matrix not found")

    # 3) Locks (defaults to zero matrix for colors without lock section).
    locks: Dict[int, List[List[int]]] = {}
    for _, _, color, parsed in lock_entries:
        if len(parsed) == rows and all(len(r) == cols for r in parsed):
            locks[color] = parsed

    for color in row_targets:
        if color not in locks:
            locks[color] = [[0 for _ in range(cols)] for _ in range(rows)]

    # 4) Blocks: each 'colour = n' followed by next matrix.
    blocks: List[Block] = []
    for i, m in enumerate(re.finditer(r"(\S*?\d+)\s*:\s*\{\s*colour\s*=\s*(\d+)\s*;", raw, flags=re.S)):
        block_id = m.group(1)
        color = int(m.group(2))
        idx = m.end()
        while idx < len(raw) and raw[idx] != "[":
            idx += 1
        if idx >= len(raw):
            raise ValueError(f"Block {block_id} missing shape matrix")
        literal, _ = _extract_bracket_expr(raw, idx)
        shape = _clean_list_literal(literal)
        if not _is_matrix(shape):
            raise ValueError(f"Block {block_id} has invalid shape matrix")
        blocks.append(Block(block_id=f"block_{i+1}", color=color, shape=shape))

    if not blocks:
        raise ValueError("No blocks parsed")

    return Puzzle(
        rows=rows,
        cols=cols,
        row_targets=row_targets,
        col_targets=col_targets,
        gray=gray,
        locks=locks,
        blocks=blocks,
        color_rgb={},
    )


def _color_key_to_int(key: str) -> int:
    m = re.match(r"colour_(\d+)$", key)
    if not m:
        raise ValueError(f"Invalid color key: {key}")
    return int(m.group(1))


def _parse_json_puzzle(obj: dict) -> Puzzle:
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object")

    board_obj = None
    gray = None
    locks_obj = None
    color_rgb_obj = None
    blocks_raw: List[Tuple[str, object]] = []

    for k, v in obj.items():
        if isinstance(v, dict):
            # board: {colour_n: [[...],[...]]}
            if all(
                isinstance(kk, str)
                and re.match(r"colour_\d+$", kk)
                and isinstance(vv, list)
                and len(vv) == 2
                and _is_list_of_ints(vv[0])
                and _is_list_of_ints(vv[1])
                for kk, vv in v.items()
            ):
                board_obj = v
                continue
            # locks: {colour_n: matrix}
            if all(
                isinstance(kk, str)
                and re.match(r"colour_\d+$", kk)
                and isinstance(vv, list)
                for kk, vv in v.items()
            ):
                # RGB color map: {colour_n:[r,g,b]}
                if all(
                    isinstance(vv, list)
                    and len(vv) == 3
                    and all(isinstance(x, int) and 0 <= x <= 255 for x in vv)
                    for vv in v.values()
                ):
                    color_rgb_obj = v
                elif all(_is_matrix(vv) for vv in v.values()):
                    locks_obj = v
                continue
            # block: {"colour": "colour_n"/n, "xxx": matrix}
            if "colour" in v and any(_is_matrix(vv) for vv in v.values()):
                blocks_raw.append((k, v))
                continue
        elif _is_matrix(v):
            gray = v

    if board_obj is None:
        raise ValueError("JSON puzzle missing board hints object")

    row_targets: Dict[int, List[int]] = {}
    col_targets: Dict[int, List[int]] = {}
    for ck, vv in board_obj.items():
        color = _color_key_to_int(ck)
        col_targets[color] = list(vv[0])
        row_targets[color] = list(vv[1])

    any_color = next(iter(row_targets.keys()))
    rows = len(row_targets[any_color])
    cols = len(col_targets[any_color])
    for c in row_targets:
        if len(row_targets[c]) != rows or len(col_targets[c]) != cols:
            raise ValueError("Inconsistent board sizes across colors")

    if gray is None or len(gray) != rows or any(len(r) != cols for r in gray):
        raise ValueError("JSON puzzle missing valid gray-cell matrix")

    locks: Dict[int, List[List[int]]] = {}
    if locks_obj:
        for ck, vv in locks_obj.items():
            color = _color_key_to_int(ck)
            if len(vv) == rows and all(len(r) == cols for r in vv):
                locks[color] = vv

    for color in row_targets:
        if color not in locks:
            locks[color] = [[0 for _ in range(cols)] for _ in range(rows)]

    color_rgb: Dict[int, Tuple[int, int, int]] = {}
    if color_rgb_obj:
        for ck, vv in color_rgb_obj.items():
            color = _color_key_to_int(ck)
            color_rgb[color] = (int(vv[0]), int(vv[1]), int(vv[2]))

    blocks: List[Block] = []
    for i, (_, bv) in enumerate(sorted(blocks_raw, key=lambda x: x[0])):
        color_val = bv["colour"]
        if isinstance(color_val, str):
            color = _color_key_to_int(color_val)
        elif isinstance(color_val, int):
            color = color_val
        else:
            raise ValueError(f"Invalid block color type: {type(color_val)}")

        shape = None
        for vv in bv.values():
            if _is_matrix(vv):
                shape = vv
                break
        if shape is None:
            raise ValueError("Block missing shape matrix")
        blocks.append(Block(block_id=f"block_{i+1}", color=color, shape=shape))

    if not blocks:
        raise ValueError("No blocks parsed from JSON puzzle")

    return Puzzle(
        rows=rows,
        cols=cols,
        row_targets=row_targets,
        col_targets=col_targets,
        gray=gray,
        locks=locks,
        blocks=blocks,
        color_rgb=color_rgb,
    )


def parse_puzzle(path: str) -> Puzzle:
    raw = None
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            with open(path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError(f"Cannot decode file: {path}")

    # Prefer JSON parser for *.json, fallback to tolerant text parser.
    if path.lower().endswith(".json"):
        try:
            return _parse_json_puzzle(json.loads(raw))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}") from e

    return _parse_text_puzzle(raw)


def normalize_cells(cells: Sequence[Coord]) -> Tuple[Coord, ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(norm)


def rotate90(cells: Sequence[Coord]) -> List[Coord]:
    # (r, c) -> (c, -r)
    return [(c, -r) for r, c in cells]


def make_orientations(block: Block) -> List[OrientedBlock]:
    base = [(r, c) for r, row in enumerate(block.shape) for c, v in enumerate(row) if v == 1]
    if not base:
        raise ValueError(f"Empty block shape: {block.block_id}")

    seen: Set[Tuple[Coord, ...]] = set()
    out: List[OrientedBlock] = []

    current = base
    for _ in range(4):
        norm = normalize_cells(current)
        if norm not in seen:
            seen.add(norm)
            h = max(r for r, _ in norm) + 1
            w = max(c for _, c in norm) + 1
            out.append(
                OrientedBlock(
                    block_id=block.block_id,
                    color=block.color,
                    cells=list(norm),
                    height=h,
                    width=w,
                )
            )
        current = rotate90(current)
    return out


def solve_puzzle(puzzle: Puzzle):
    rows, cols = puzzle.rows, puzzle.cols
    colors = sorted(puzzle.row_targets.keys())

    blocked = [[False for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if puzzle.gray[r][c] == 1:
                blocked[r][c] = True

    for color in colors:
        for r in range(rows):
            for c in range(cols):
                if puzzle.locks[color][r][c] == 1:
                    blocked[r][c] = True

    occupancy = [[0 for _ in range(cols)] for _ in range(rows)]
    block_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    row_used: Dict[int, List[int]] = {c: [0] * rows for c in colors}
    col_used: Dict[int, List[int]] = {c: [0] * cols for c in colors}

    # Locked cells count toward targets.
    for color in colors:
        for r in range(rows):
            for c in range(cols):
                if puzzle.locks[color][r][c] == 1:
                    row_used[color][r] += 1
                    col_used[color][c] += 1

    # Fast sanity checks.
    for color in colors:
        if any(row_used[color][r] > puzzle.row_targets[color][r] for r in range(rows)):
            raise ValueError(f"Color {color}: locked cells exceed row target")
        if any(col_used[color][c] > puzzle.col_targets[color][c] for c in range(cols)):
            raise ValueError(f"Color {color}: locked cells exceed column target")

    area_by_color: Dict[int, int] = {c: 0 for c in colors}
    for b in puzzle.blocks:
        area_by_color[b.color] += sum(sum(row) for row in b.shape)

    for color in colors:
        row_sum = sum(puzzle.row_targets[color])
        col_sum = sum(puzzle.col_targets[color])
        if row_sum != col_sum:
            raise ValueError(
                f"Color {color}: row target sum ({row_sum}) != column target sum ({col_sum}); puzzle data is inconsistent"
            )
        need = sum(puzzle.row_targets[color]) - sum(row_used[color])
        if need != area_by_color[color]:
            raise ValueError(
                f"Color {color}: required fill cells ({need}) != total block cells ({area_by_color[color]})"
            )

    oriented = [make_orientations(b) for b in puzzle.blocks]
    placements: List[Tuple[int, int, OrientedBlock]] = []

    def can_place(ob: OrientedBlock, r0: int, c0: int) -> bool:
        color = ob.color
        delta_row: Dict[int, int] = {}
        delta_col: Dict[int, int] = {}

        for dr, dc in ob.cells:
            r = r0 + dr
            c = c0 + dc
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return False
            if blocked[r][c] or occupancy[r][c] != 0:
                return False
            delta_row[r] = delta_row.get(r, 0) + 1
            delta_col[c] = delta_col.get(c, 0) + 1

        for r, d in delta_row.items():
            if row_used[color][r] + d > puzzle.row_targets[color][r]:
                return False
        for c, d in delta_col.items():
            if col_used[color][c] + d > puzzle.col_targets[color][c]:
                return False
        return True

    def place(ob: OrientedBlock, r0: int, c0: int, block_tag: int):
        color = ob.color
        for dr, dc in ob.cells:
            r = r0 + dr
            c = c0 + dc
            occupancy[r][c] = color
            block_grid[r][c] = block_tag
            row_used[color][r] += 1
            col_used[color][c] += 1

    def unplace(ob: OrientedBlock, r0: int, c0: int):
        color = ob.color
        for dr, dc in ob.cells:
            r = r0 + dr
            c = c0 + dc
            occupancy[r][c] = 0
            block_grid[r][c] = 0
            row_used[color][r] -= 1
            col_used[color][c] -= 1

    def final_ok() -> bool:
        for color in colors:
            if row_used[color] != puzzle.row_targets[color]:
                return False
            if col_used[color] != puzzle.col_targets[color]:
                return False
        return True

    def backtrack(i: int) -> bool:
        if i == len(oriented):
            return final_ok()

        # Simple fail-first: try blocks with fewer candidate states first by local reordering.
        # Keep implementation cheap: evaluate current block only.
        for ob in oriented[i]:
            for r0 in range(rows - ob.height + 1):
                for c0 in range(cols - ob.width + 1):
                    if not can_place(ob, r0, c0):
                        continue
                    place(ob, r0, c0, i + 1)
                    placements.append((r0, c0, ob))
                    if backtrack(i + 1):
                        return True
                    placements.pop()
                    unplace(ob, r0, c0)
        return False

    if not backtrack(0):
        return None

    return {
        "occupancy": occupancy,
        "block_grid": block_grid,
        "placements": placements,
        "row_used": row_used,
        "col_used": col_used,
    }


def save_solution_image(puzzle: Puzzle, occupancy: List[List[int]], block_grid: List[List[int]], out_path: str):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as e:
        raise RuntimeError("matplotlib is required for image output. Please install it: pip install matplotlib") from e

    rows, cols = puzzle.rows, puzzle.cols
    colors = sorted(puzzle.row_targets.keys())

    # Distinct, readable palette for up to several colors.
    palette = {
        1: "#4E79A7",
        2: "#F28E2B",
        3: "#E15759",
        4: "#76B7B2",
        5: "#59A14F",
    }
    for color, (r, g, b) in puzzle.color_rgb.items():
        palette[color] = f"#{r:02X}{g:02X}{b:02X}"

    fig_w = max(6, cols * 0.8)
    fig_h = max(6, rows * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for r in range(rows):
        for c in range(cols):
            y = rows - 1 - r
            face = "#FFFFFF"
            edge = "#BBBBBB"
            hatch = None

            if puzzle.gray[r][c] == 1:
                face = "#666666"
                edge = "#444444"
            else:
                occ = occupancy[r][c]
                if occ != 0:
                    face = palette.get(occ, "#999999")
                    edge = "#222222"
                else:
                    lock_color = 0
                    for color in colors:
                        if puzzle.locks[color][r][c] == 1:
                            lock_color = color
                            break
                    if lock_color:
                        face = palette.get(lock_color, "#BBBBBB")
                        edge = "#333333"
                        hatch = "xx"

            rect = Rectangle((c, y), 1, 1, facecolor=face, edgecolor=edge, linewidth=2, hatch=hatch)
            ax.add_patch(rect)

    # Thick boundaries around each block so same-color adjacent blocks are distinguishable.
    boundary_color = "#111111"
    boundary_width = 4.8
    for r in range(rows):
        for c in range(cols):
            bid = block_grid[r][c]
            if bid == 0:
                continue
            y0 = rows - 1 - r
            # Top
            if r == 0 or block_grid[r - 1][c] != bid:
                ax.plot([c, c + 1], [y0 + 1, y0 + 1], color=boundary_color, linewidth=boundary_width)
            # Bottom
            if r == rows - 1 or block_grid[r + 1][c] != bid:
                ax.plot([c, c + 1], [y0, y0], color=boundary_color, linewidth=boundary_width)
            # Left
            if c == 0 or block_grid[r][c - 1] != bid:
                ax.plot([c, c], [y0, y0 + 1], color=boundary_color, linewidth=boundary_width)
            # Right
            if c == cols - 1 or block_grid[r][c + 1] != bid:
                ax.plot([c + 1, c + 1], [y0, y0 + 1], color=boundary_color, linewidth=boundary_width)

    # Grid lines
    for c in range(cols + 1):
        ax.plot([c, c], [0, rows], color="#222222", linewidth=0.8)
    for r in range(rows + 1):
        ax.plot([0, cols], [r, r], color="#222222", linewidth=0.8)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Arknights: Endfield puzzle solution", fontsize=12)

    # Legend text
    legend_lines = ["Filled: block color", "Hatched: locked colored cell", "Dark gray: unavailable cell"]
    fig.text(0.02, 0.02, "\n".join(legend_lines), fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Solve Arknights: Endfield puzzle from example text format.")
    parser.add_argument("input", help="Path to puzzle txt file")
    parser.add_argument("--out", default=None, help="Output image path (PNG)")
    args = parser.parse_args()

    puzzle = parse_puzzle(args.input)
    try:
        result = solve_puzzle(puzzle)
    except ValueError as e:
        print(f"Invalid/unsatisfiable puzzle data: {e}")
        return 1

    if result is None:
        print("No solution found.")
        return 1

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join("outputs", f"{base}_solution.png")

    save_solution_image(puzzle, result["occupancy"], result["block_grid"], out_path)
    print(f"Solved. Output image: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
