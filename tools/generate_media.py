"""Generate the images and video under ``docs/assets/``.

A maintainer tool, not part of the package. Nothing it produces is imported at
runtime, and nothing it needs is a runtime dependency: the engine still
installs with nothing but the standard library. Pillow is used here and only
here, and ffmpeg only to encode.

    python tools/generate_media.py all
    python tools/generate_media.py boards charts
    python tools/generate_media.py selfplay --plies 40 --depth 3

The committed files are the artefact; this script is here so they are
reproducible rather than mystery binaries. Regenerating needs Pillow, ffmpeg on
the PATH, and a font with real chess glyphs. That font is used only at
generation time and is deliberately not vendored -- set ``FOXCHESS_PIECE_FONT``
to a ``.ttf`` path if the search below cannot find one.

Boards are drawn with a1 at the bottom left, matching the engine's own square
numbering, so a square in a picture and a square in a FEN are the same square.

Every video frame is 1280x720. Video encoders require one frame size for the
whole file, so the search animation draws its score trace as an inset in the
same panel rather than as separate full-frame charts.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "Pillow is required to generate the images.\n"
        "    pip install -e '.[media]'"
    ) from None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from foxchess import notation  # noqa: E402
from foxchess.board import Board  # noqa: E402
from foxchess.evaluate import evaluate  # noqa: E402
from foxchess.move import NO_MOVE, decode_from, decode_to, from_uci  # noqa: E402
from foxchess.search import MATE_SCORE, Searcher, SearchLimits  # noqa: E402
from foxchess.types import Color, PieceType, color_of, piece_type_of  # noqa: E402
from foxchess.uci import UciSession  # noqa: E402

ASSETS = REPO_ROOT / "docs" / "assets"

#: Chess symbols, keyed by (colour, piece type) and spelled out rather than
#: indexed off an ordering. ``PieceType`` runs pawn, knight, bishop, rook, queen,
#: king -- the opposite of the Unicode block, whose filled pieces are ordered
#: king first. Computing the index arithmetically renders a board of entirely
#: incorrect pieces, and looks plausible enough to ship, so the mapping is
#: written out.
PIECE_GLYPHS: dict[tuple[Color, PieceType], str] = {
    (Color.WHITE, PieceType.PAWN): "\u2659",
    (Color.WHITE, PieceType.KNIGHT): "\u2658",
    (Color.WHITE, PieceType.BISHOP): "\u2657",
    (Color.WHITE, PieceType.ROOK): "\u2656",
    (Color.WHITE, PieceType.QUEEN): "\u2655",
    (Color.WHITE, PieceType.KING): "\u2654",
    (Color.BLACK, PieceType.PAWN): "\u265F",
    (Color.BLACK, PieceType.KNIGHT): "\u265E",
    (Color.BLACK, PieceType.BISHOP): "\u265D",
    (Color.BLACK, PieceType.ROOK): "\u265C",
    (Color.BLACK, PieceType.QUEEN): "\u265B",
    (Color.BLACK, PieceType.KING): "\u265A",
}

#: Private-use codepoints, used to discover what a font draws when it has
#: nothing: its .notdef box. Without this, a font missing every chess character
#: still renders twelve identical boxes, which sail through any test that only
#: asks "did it draw something".
_NOTDEF_PROBE = "\uE123\uE124\uE125"

PIECE_FONT_ENV = "FOXCHESS_PIECE_FONT"

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\seguisym.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
)

FRAME = (1280, 720)

#: Centipawns a mate is plotted at, so a real mate score does not swamp the
#: axis. Deep enough to read as decisive, shallow enough to leave the earlier
#: iterations of the trace visible.
MATE_DISPLAY = 1500

#: (regular, bold) pairs for text. Labels need no chess coverage, only
#: something legible, so these are tried in order and the first hit wins.
_UI_FONT_CANDIDATES = (
    (
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
    ),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
)


@dataclass(frozen=True)
class Palette:
    """Every colour in one place, so the assets stay visually consistent."""

    light: tuple[int, int, int] = (240, 217, 181)
    dark: tuple[int, int, int] = (181, 136, 99)
    page: tuple[int, int, int] = (22, 24, 28)
    panel: tuple[int, int, int] = (31, 34, 40)
    text: tuple[int, int, int] = (236, 238, 242)
    muted: tuple[int, int, int] = (138, 146, 158)
    accent: tuple[int, int, int] = (109, 178, 224)
    good: tuple[int, int, int] = (126, 200, 140)
    bad: tuple[int, int, int] = (222, 118, 118)
    grid: tuple[int, int, int] = (56, 60, 68)


DARK = Palette()


# ---------------------------------------------------------------------------
# Fonts


def _font(path: Path, size: int):
    return ImageFont.truetype(str(path), size)


def _signature(font, text: str) -> set[tuple]:
    return {(font.getmask(ch).size, bytes(font.getmask(ch))) for ch in text}


def _covers_chess(path: Path) -> bool:
    """True when the font really carries all twelve chess glyphs.

    Rendering them is not sufficient evidence: a font with no coverage draws
    twelve identical .notdef boxes. So the glyphs are compared against what the
    same font draws for codepoints it certainly lacks, and all twelve must
    differ from that.
    """
    try:
        probe = _font(path, 48)
    except OSError:
        return False
    wanted = "".join(PIECE_GLYPHS.values())
    notdef = _signature(probe, _NOTDEF_PROBE)
    drawn = _signature(probe, wanted)
    return len(drawn) == len(PIECE_GLYPHS) and not (drawn & notdef)


def _candidates() -> list[Path]:
    override = os.environ.get(PIECE_FONT_ENV)
    found = [Path(override)] if override else []
    return found + [Path(p) for p in _FONT_CANDIDATES]


def piece_font(size: int):
    """The first candidate font that genuinely has chess glyphs."""
    for candidate in _candidates():
        if candidate.is_file() and _covers_chess(candidate):
            return _font(candidate, size)
    searched = "\n    ".join(str(p) for p in _candidates())
    raise SystemExit(
        "No font with chess glyphs was found.\n"
        f"    Looked at:\n    {searched}\n"
        "  Install Segoe UI Symbol, DejaVu Sans, FreeSerif or Noto Sans Symbols 2,\n"
        f"  or point {PIECE_FONT_ENV} at a .ttf that has them."
    )


def ui_font(size: int, bold: bool = False):
    """A font for labels and numbers, which need no chess coverage."""
    override = os.environ.get(PIECE_FONT_ENV)
    pairs = [(override, override)] if override and Path(override).is_file() else []
    pairs += list(_UI_FONT_CANDIDATES)
    for regular, heavy in pairs:
        chosen = heavy if bold else regular
        if chosen and Path(chosen).is_file():
            return _font(Path(chosen), size)
    # The piece font is Segoe UI Symbol, which is also a perfectly readable text
    # font, so a missing UI font is never a reason to give up.
    return piece_font(size)


# ---------------------------------------------------------------------------
# Drawing


def _glyph_for(piece: int) -> str:
    return PIECE_GLYPHS[(color_of(piece), piece_type_of(piece))]


def _cell(sq: int, square: int, origin: tuple[int, int]) -> tuple[int, int]:
    """Top-left pixel of a square's cell, with rank 1 along the bottom."""
    return origin[0] + (sq & 7) * square, origin[1] + (7 - (sq >> 3)) * square


def draw_board(
    draw,
    board: Board,
    *,
    origin: tuple[int, int],
    square: int,
    palette: Palette,
    piece,
    coords: bool = False,
    from_sq: int | None = None,
    to_sq: int | None = None,
) -> None:
    for sq in range(64):
        x, y = _cell(sq, square, origin)
        tone = palette.light if ((sq & 7) + (sq >> 3)) % 2 == 0 else palette.dark
        draw.rectangle((x, y, x + square - 1, y + square - 1), fill=tone)

    for sq, piece_code in enumerate(board.mailbox):
        if not piece_code:
            continue
        x, y = _cell(sq, square, origin)
        glyph = _glyph_for(piece_code)
        left, top, right, bottom = draw.textbbox((0, 0), glyph, font=piece)
        draw.text(
            (x + (square - (right - left)) / 2 - left, y + (square - (bottom - top)) / 2 - top),
            glyph,
            font=piece,
            fill=palette.page,
        )

    for marked in (from_sq, to_sq):
        if marked is None:
            continue
        x, y = _cell(marked, square, origin)
        draw.rectangle(
            (x + 3, y + 3, x + square - 4, y + square - 4),
            outline=palette.accent,
            width=max(3, square // 14),
        )

    if coords:
        label = ui_font(max(11, square // 5))
        for index in range(8):
            x, _ = _cell(index, square, origin)
            draw.text(
                (x + square / 2, origin[1] + square * 8 + 4),
                "abcdefgh"[index],
                font=label,
                fill=palette.muted,
                anchor="mt",
            )
            _, y = _cell(index * 8, square, origin)
            draw.text(
                (origin[0] - 5, y + square / 2),
                "abcdefgh"[index],
                font=label,
                fill=palette.muted,
                anchor="rm",
            )


def _score(board: Board, side: Color) -> str:
    centipawns = int(evaluate(board, detailed=False))
    if side is Color.BLACK:
        centipawns = -centipawns
    if abs(centipawns) >= MATE_SCORE - 1000:
        moves = (MATE_SCORE - abs(centipawns) + 1) // 2
        return f"#{moves}" if centipawns > 0 else f"-#{moves}"
    return f"{centipawns / 100:+.2f}"


def _status(board: Board) -> str:
    if board.is_checkmate():
        return "Checkmate"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Draw by insufficient material"
    if board.is_game_over():
        return "Game over"
    return "White to move" if board.side is Color.WHITE else "Black to move"


def _panel(
    board: Board,
    *,
    title: str,
    subtitle: str = "",
    last_move: tuple[int, int] | None = None,
    size: tuple[int, int] = FRAME,
    palette: Palette = DARK,
    trace: list[tuple[int, int]] | None = None,
    highlight_depth: int | None = None,
) -> object:
    """A board on the left, a readout on the right. Every video frame uses this."""
    width, height = size
    image = Image.new("RGB", size, palette.panel)
    draw = ImageDraw.Draw(image)

    square = 66 if height <= 720 else 84
    margin = 34
    pad = 30
    board_px = square * 8 + pad * 2
    top = max(margin, (height - board_px) // 2)
    draw_board(
        draw,
        board,
        origin=(margin + pad, top + pad),
        square=square,
        palette=palette,
        piece=piece_font(int(square * 0.8)),
    )

    x = margin + board_px + 26
    y = 44
    heading = ui_font(26, bold=True)
    body = ui_font(17)
    small = ui_font(14)

    draw.text((x, y), title, font=heading, fill=palette.text)
    y += 34
    if subtitle:
        draw.text((x, y), subtitle, font=small, fill=palette.muted)
        y += 22
    y += 14

    white = _score(board, Color.WHITE)
    black = _score(board, Color.BLACK)
    decisive = abs(evaluate(board, detailed=False)) >= 20
    white_leads = not white.startswith("-")
    for name, value, leads in (("White", white, white_leads), ("Black", black, not white_leads)):
        tone = palette.text if name == "White" else palette.muted
        if leads and decisive:
            tone = palette.good
        draw.text((x, y), name, font=body, fill=tone)
        draw.text((x + 92, y), value, font=body, fill=tone, anchor="ra")
        y += 25

    y += 10
    draw.line((x, y, width - 40, y), fill=palette.grid, width=1)
    y += 14
    draw.text((x, y), _status(board), font=body, fill=palette.text)
    y += 32

    if trace:
        draw.text((x, y), "SCORE BY DEPTH", font=small, fill=palette.muted)
        y += 20
        _draw_trace(draw, (x, y, width - 40, y + 96), trace, highlight_depth, palette)
        y += 112
        if last_move:
            draw.text((x, y), "LINE", font=small, fill=palette.muted)
            y += 19
    else:
        draw.text((x, y), "FEN", font=small, fill=palette.muted)
        y += 19

    fen = board.fen()
    for start in range(0, len(fen), 46):
        draw.text((x, y), fen[start : start + 46], font=small, fill=palette.muted)
        y += 18

    return image


def _draw_trace(draw, box, trace, highlight, palette: Palette) -> None:
    """A score-vs-depth line, with zero as a marked reference.

    The vertical scale is clamped rather than fitted, so a wild early score does
    not squash the later iterations into a flat line -- the interesting part of
    the curve is the part that settles.
    """
    left, top, right, bottom = box
    mid = (top + bottom) / 2
    draw.line((left, mid, right, mid), fill=palette.grid, width=1)
    draw.text((left - 4, mid), "0", font=ui_font(11), fill=palette.muted, anchor="rm")

    span = max(100.0, max(abs(score) for _, score in trace) * 1.2)
    points = []
    for index, (_depth, score) in enumerate(trace):
        px = left + index / max(len(trace) - 1, 1) * (right - left)
        py = mid - max(-1.0, min(1.0, score / span)) * (bottom - top) / 2
        points.append((px, py, index))

    if len(points) > 1:
        draw.line([(px, py) for px, py, _ in points], fill=palette.accent, width=2)
    for px, py, index in points:
        done = highlight is not None and trace[index][0] <= highlight
        radius = 4 if done else 3
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=palette.accent if done else palette.muted,
        )


# ---------------------------------------------------------------------------
# Still images


def _show_positions() -> list[tuple[str, str, str]]:
    return [
        (
            "startpos",
            "The starting position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "kiwipete",
            "Kiwipete: the perft workhorse",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
        (
            "en-passant",
            "En passant available",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ),
        (
            "stalemate",
            "Stalemate: no legal move, not in check",
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        ),
    ]


def generate_boards() -> list[Path]:
    made = []
    for name, caption, fen in _show_positions():
        size = 520
        pad = 26
        image = Image.new("RGB", (size, size + 40), DARK.page)
        draw = ImageDraw.Draw(image)
        draw_board(
            draw,
            Board(fen),
            origin=(pad, pad),
            square=(size - pad * 2) // 8,
            palette=DARK,
            piece=piece_font(46),
            coords=True,
        )
        draw.text((size // 2, size + 8), caption, font=ui_font(14), fill=DARK.muted, anchor="mt")
        target = ASSETS / f"board-{name}.png"
        image.save(target)
        made.append(target)
        print(f"  {target.name}  {caption}")
    return made


def _perft_data() -> list[tuple[str, list[int]]]:
    """The perft suite, read from the tests rather than restated here.

    An earlier draft of this file carried its own copy of these numbers and got
    one of them wrong. The counts are the whole point of the chart, so they are
    loaded from ``tests/conftest.py`` -- the same source the slow suite asserts
    against -- which makes drift impossible rather than merely unlikely.
    """
    path = REPO_ROOT / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_foxchess_perft", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load the perft suite from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [(name, list(counts)) for name, _fen, counts in module.PERFT_SUITE]


def generate_perft_chart() -> Path:
    data = _perft_data()
    width, height = 1040, 640
    left, top, right, bottom = 92, 84, width - 150, height - 78
    image = Image.new("RGB", (width, height), DARK.panel)
    draw = ImageDraw.Draw(image)

    draw.text((left, 26), "Perft leaf nodes by depth", font=ui_font(24, bold=True), fill=DARK.text)
    draw.text(
        (left, 56),
        "Published counts, the same ones the slow suite asserts",
        font=ui_font(14),
        fill=DARK.muted,
    )

    deepest = max(len(counts) for _, counts in data)
    largest = max(counts[-1] for _, counts in data)
    lo, hi = 0.0, math.ceil(math.log10(largest))

    def y_of(nodes: int) -> float:
        return bottom - (math.log10(max(nodes, 1)) - lo) / (hi - lo) * (bottom - top)

    def x_of(depth: int) -> float:
        return left + (depth - 1) / (deepest - 1) * (right - left)

    for exponent in range(int(hi) + 1):
        value = 10**exponent
        y = y_of(value)
        draw.line((left, y, right, y), fill=DARK.grid, width=1)
        draw.text((left - 10, y), f"{value:,}", font=ui_font(12), fill=DARK.muted, anchor="rm")

    for depth in range(1, deepest + 1):
        x = x_of(depth)
        draw.line((x, top, x, bottom), fill=DARK.grid, width=1)
        draw.text((x, bottom + 8), str(depth), font=ui_font(14), fill=DARK.muted, anchor="mt")

    draw.text((left, height - 30), "depth", font=ui_font(13), fill=DARK.muted)

    colours = [
        DARK.accent,
        DARK.good,
        (214, 168, 106),
        DARK.bad,
        (176, 140, 208),
        (110, 200, 200),
    ]
    for (name, counts), colour in zip(data, colours, strict=True):
        points = [(x_of(depth), y_of(nodes)) for depth, nodes in enumerate(counts, 1)]
        draw.line(points, fill=colour, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
        draw.text(
            (points[-1][0] + 12, points[-1][1]),
            name,
            font=ui_font(13),
            fill=colour,
            anchor="lm",
        )

    target = ASSETS / "chart-perft.png"
    image.save(target)
    print(f"  {target.name}")
    return target


def generate_nps_chart() -> Path:
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    measured = []
    for depth in (2, 3, 4):
        result = Searcher(Board(fen)).search(SearchLimits(depth=depth))
        measured.append((depth, result.nodes, result.nps))
        print(f"  measured depth {depth}: {result.nps:,} nps over {result.nodes:,} nodes")

    width, height = 1040, 500
    left, top, right, bottom = 90, 84, width - 60, height - 92
    image = Image.new("RGB", (width, height), DARK.panel)
    draw = ImageDraw.Draw(image)
    draw.text((left, 26), "Search throughput by depth", font=ui_font(24, bold=True), fill=DARK.text)
    draw.text(
        (left, 56),
        "Kiwipete, measured on whichever machine generated this file",
        font=ui_font(14),
        fill=DARK.muted,
    )

    ceiling = max(nps for _, _, nps in measured) * 1.2
    for index, (depth, nodes, nps) in enumerate(measured):
        x = left + 30 + index * 190
        bar = max(4, nps / ceiling * (bottom - top))
        draw.rectangle((x, bottom - bar, x + 96, bottom), fill=DARK.accent)
        draw.text(
            (x + 48, bottom - bar - 10),
            f"{nps:,} nps",
            font=ui_font(16, bold=True),
            fill=DARK.text,
            anchor="mb",
        )
        draw.text(
            (x + 48, bottom - bar + 10),
            f"{nodes:,} nodes",
            font=ui_font(12),
            fill=DARK.page,
            anchor="mt",
        )
        draw.text(
            (x + 48, bottom + 10),
            f"depth {depth}",
            font=ui_font(14),
            fill=DARK.muted,
            anchor="mt",
        )

    draw.line((left, bottom, right, bottom), fill=DARK.grid, width=1)
    target = ASSETS / "chart-nps.png"
    image.save(target)
    print(f"  {target.name}")
    return target


# ---------------------------------------------------------------------------
# Video


def _encode(frames: list, fps: int, name: str) -> list[Path]:
    """Encode one frame sequence to both an MP4 and a GIF.

    ``name`` is the shared stem: the MP4 and GIF are two encodings of the same
    frames, which is why they are written together and cannot drift apart.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on the PATH, so the videos cannot be encoded")
    made: list[Path] = []
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        for index, frame in enumerate(frames):
            frame.save(directory / f"frame_{index:04d}.png")
        pattern = str(directory / "frame_%04d.png")

        mp4 = ASSETS / f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "22",
                str(mp4),
            ],
            check=True,
            capture_output=True,
        )
        made.append(mp4)

        palette = directory / "palette.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-vf",
                f"fps={fps},scale=760:-2:flags=lanczos,palettegen=max_colors=192",
                str(palette),
            ],
            check=True,
            capture_output=True,
        )
        gif = ASSETS / f"{name}.gif"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-i",
                str(palette),
                "-lavfi",
                f"fps={fps},scale=760:-2:flags=lanczos[s];[s][1:v]paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                str(gif),
            ],
            check=True,
            capture_output=True,
        )
        made.append(gif)

    for path in made:
        print(f"  {path.name}  {path.stat().st_size / 1024:,.0f} KB")
    return made


def _hold(image, frames: int) -> list:
    return [image] * frames


def generate_selfplay(plies: int, depth: int, fps: int, hold: int) -> list[Path]:
    """A real game, searched move by move, as a board animation.

    Not the CLI's ``pgn`` demo: that picks ``min(legal_moves)`` so its output is
    reproducible, which makes for a dull game. Here every move comes out of a
    real search, so the animation shows the engine thinking rather than a
    scripted position list.
    """
    board = Board()
    intro = _panel(
        board,
        title="foxchess, playing itself",
        subtitle=f"every move chosen by a depth-{depth} search",
    )
    frames: list = _hold(intro, hold * 2)
    played = 0

    for number in range(plies):
        if board.is_game_over():
            break
        result = Searcher(board).search(SearchLimits(depth=depth))
        if result.best_move == NO_MOVE:
            break
        text = notation.san(board, result.best_move)
        origin, target = decode_from(result.best_move), decode_to(result.best_move)
        board.make_move(result.best_move)
        played += 1
        frames += _hold(
            _panel(
                board,
                title="foxchess, playing itself",
                subtitle=(
                    f"move {number // 2 + 1}  {text}   depth {result.depth}   "
                    f"{result.nodes:,} nodes   {result.nps:,} nps"
                ),
                last_move=(origin, target),
            ),
            hold,
        )
        print(f"  {number // 2 + 1:>2}. {text:<8} eval {_score(board, board.side)}")

    frames += _hold(
        _panel(
            board,
            title="foxchess, playing itself",
            subtitle=f"{board.result()} after {played} half-moves",
        ),
        hold * 3,
    )
    return _encode(frames, fps, "selfplay")


def generate_search(fen: str, depth: int, fps: int, hold: int) -> list[Path]:
    """Animate one search, with a score trace drawn as an inset.

    Depth, score and the principal variation all come from the engine's own UCI
    ``info`` lines, so a frame cannot claim anything the engine did not report.
    Node counts are deliberately absent: ``Searcher.on_update`` is
    ``(depth, score, pv)``, so the engine never reports them mid-search, and
    inventing a number here would be a lie in a screenshot.
    """
    board = Board(fen)
    stream = io.StringIO()
    session = UciSession(out=stream)
    for _ in session.handle(f"position fen {board.fen()}"):
        pass

    # ``info`` is written to the session's output stream as each depth lands;
    # ``handle`` returns only the final ``bestmove``. Reading the stream is how
    # the numbers here are the engine's own rather than this script's guesswork.
    for _ in session.handle(f"go depth {depth}"):
        pass

    series: list[tuple[int, int, list[str], int | None]] = []
    for line in stream.getvalue().splitlines():
        if not line.startswith("info depth"):
            continue
        fields = line.split()
        mate: int | None = None
        if "mate" in fields:
            mate = int(fields[fields.index("mate") + 1])
            # A mate is MATE_SCORE in centipawns, which is ~30000. Plotted
            # literally it flattens every earlier iteration into the axis, so it
            # is clamped for the trace and reported as a distance instead.
            score = MATE_DISPLAY * (1 if mate > 0 else -1)
        else:
            score = int(fields[fields.index("cp") + 1])
        pv = line.split(" pv ")[1].split() if " pv " in line else []
        series.append((int(fields[2]), score, pv, mate))
    if not series:
        raise SystemExit("the search reported no info lines, so there is nothing to animate")

    trace = [(d, s) for d, s, _, _ in series]
    frames: list = []
    for reached, score, pv, mate in series:
        shown = board
        highlighted = None
        if pv:
            first = from_uci(pv[0], board.legal_moves())
            if first is not None:
                highlighted = (decode_from(first), decode_to(first))
                shown = _after(board, first)
        if mate is not None:
            detail = f"mate in {abs(mate)}" if mate > 0 else f"mated in {abs(mate)}"
        else:
            detail = f"{len(pv)} moves in the principal variation" if pv else "no line yet"
        frames += _hold(
            _panel(
                shown,
                title=f"searching, depth {reached}",
                subtitle=detail,
                last_move=highlighted,
                trace=trace,
                highlight_depth=reached,
            ),
            hold,
        )
        shown_score = f"mate in {abs(mate)}" if mate is not None else f"{score / 100:+.2f}"
        print(f"  depth {reached:>2}  {shown_score:>10}  pv {' '.join(pv[:6])}")

    best = series[-1]
    frames += _hold(
        _panel(
            board,
            title="search finished",
            subtitle=(
                f"depth {best[0]}, best line {' '.join(best[2][:8])}"
                if best[2]
                else f"depth {best[0]}, no principal variation"
            ),
        ),
        hold * 3,
    )
    return _encode(frames, fps, "search")


def _after(board: Board, move: int) -> Board:
    clone = Board(board.fen())
    clone.make_move(move)
    return clone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        default=["all"],
        choices=["all", "boards", "charts", "selfplay", "search"],
        help="which assets to build (default: all)",
    )
    parser.add_argument("--plies", type=int, default=60, help="self-play game length")
    parser.add_argument("--depth", type=int, default=4, help="search depth for both videos")
    parser.add_argument("--fps", type=int, default=12, help="animation frame rate")
    parser.add_argument("--hold", type=int, default=10, help="frames to hold each position")
    parser.add_argument(
        "--fen",
        default="1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
        help="position to animate in the search video",
    )
    args = parser.parse_args(argv)

    every = {"boards", "charts", "selfplay", "search"}
    targets = set(args.targets) | (every if "all" in args.targets else set())
    ASSETS.mkdir(parents=True, exist_ok=True)

    if "boards" in targets:
        print("boards")
        generate_boards()
    if "charts" in targets:
        print("charts")
        generate_perft_chart()
        generate_nps_chart()
    if "selfplay" in targets:
        print("selfplay")
        generate_selfplay(args.plies, args.depth, args.fps, args.hold)
    if "search" in targets:
        print("search")
        generate_search(args.fen, args.depth, args.fps, args.hold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
