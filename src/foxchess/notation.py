"""Notation: Standard Algebraic Notation, long algebraic, PGN and repetition.

SAN is the interesting part. Producing it needs *disambiguation* — deciding
whether ``Nf3`` is unambiguous or the file/rank must be added — which means
looking at every other legal move sharing the destination square. Getting that
wrong is how engines end up printing ``Rad1`` when the rules require ``R1d1``, so
the rules live in one function with the tests aimed straight at it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

from .board import STARTING_FEN, Board, IllegalMoveError
from .move import (
    CASTLE_KINGSIDE,
    CASTLE_QUEENSIDE,
    PROMOTION_PIECE,
    decode_flag,
    decode_from,
    decode_to,
    encode_int,
    from_uci,
    to_uci,
)
from .types import (
    EMPTY,
    Color,
    PieceType,
    file_of,
    name_of,
    piece_letter,
    piece_type_of,
    rank_of,
    square_from_name,
)

CAPTURE_FLAGS: Final[frozenset[int]] = frozenset({2, 3, 12, 13, 14, 15})
PROMOTION_LETTER: Final[dict[int, str]] = {
    8: "N", 9: "B", 10: "R", 11: "Q", 12: "N", 13: "B", 14: "R", 15: "Q",
}
#: SAN as it is actually written, including the ``x`` of a capture: ``Nf3``,
#: ``Nbd2``, ``exd5``, ``R1xd5``, ``e8=Q+``. The disambiguation is split into
#: separate optional file and rank groups; as one combined ``[a-h]?[1-8]?`` group
#: the regex engine cannot tell a disambiguator from the start of the target.
_SAN_BODY: Final = re.compile(
    r"^(?P<piece>[KQRBN])?(?P<file>[a-h])?(?P<rank>[1-8])?(?P<capture>x)?"
    r"(?P<target>[a-h][1-8])(?P<promotion>=?[QRBNqrbn])?$"
)
_PIECE_CHARS: Final[dict[str, PieceType]] = {
    "K": PieceType.KING,
    "Q": PieceType.QUEEN,
    "R": PieceType.ROOK,
    "B": PieceType.BISHOP,
    "N": PieceType.KNIGHT,
}
_UCI_PROMOTION: Final[dict[str, PieceType]] = {
    "q": PieceType.QUEEN, "r": PieceType.ROOK, "b": PieceType.BISHOP, "n": PieceType.KNIGHT,
}
RESULTS: Final[frozenset[str]] = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


class NotationError(IllegalMoveError):
    """Raised when a move string cannot be parsed, is ambiguous, or is not legal here.

    A notation problem *is* a kind of illegal-move complaint, so this derives from
    :class:`~foxchess.board.IllegalMoveError`: callers that already guard against
    an illegal move keep working, while a caller reading text off the network only
    has to know about this one type.
    """


# ---------------------------------------------------------------------------
# SAN output
# ---------------------------------------------------------------------------


def san(board: Board, move: int) -> str:
    """Standard Algebraic Notation for ``move``, which must be legal.

    Castling is ``O-O``/``O-O-O``, en passant is written as an ordinary capture,
    promotion uses ``=``, and ``+``/``#`` is appended based on the resulting
    position.
    """
    flag = decode_flag(move)
    from_sq = decode_from(move)
    to_sq = decode_to(move)
    capture = flag in CAPTURE_FLAGS

    if flag == CASTLE_KINGSIDE:
        text = "O-O"
    elif flag == CASTLE_QUEENSIDE:
        text = "O-O-O"
    else:
        piece = board.piece_at(from_sq)
        if piece_type_of(piece) is PieceType.PAWN:
            body = f"{name_of(from_sq)[0]}x{name_of(to_sq)}" if capture else name_of(to_sq)
        else:
            body = (
                piece_letter(piece).upper()
                + disambiguation(board, move)
                + ("x" if capture else "")
                + name_of(to_sq)
            )
        letter = PROMOTION_LETTER.get(flag)
        if letter:
            body += f"={letter}"
        text = body

    board.make_move(move)
    if board.in_check():
        text += "#" if not board.legal_moves() else "+"
    board.unmake_move()
    return text


def disambiguation(board: Board, move: int) -> str:
    """The minimal disambiguator: nothing, a file, a rank, or both.

    1. No rival reaching the same square → omit.
    2. Every rival on our file → use the rank.
    3. Every rival on our rank → use the file.
    4. Otherwise → use the file.
    """
    from_sq = decode_from(move)
    to_sq = decode_to(move)
    piece_type = piece_type_of(board.piece_at(from_sq))
    if piece_type is PieceType.KING:
        return ""

    rivals = [
        other
        for other in board.legal_moves()
        if other != move
        and decode_to(other) == to_sq
        and piece_type_of(board.piece_at(decode_from(other))) is piece_type
    ]
    if not rivals:
        return ""

    if not all(file_of(decode_from(r)) == file_of(from_sq) for r in rivals):
        return name_of(from_sq)[0]
    if not all(rank_of(decode_from(r)) == rank_of(from_sq) for r in rivals):
        return name_of(from_sq)[1]
    return name_of(from_sq)[0]


def algebraic(board: Board, move: int) -> str:  # noqa: ARG001  (signature symmetry)
    """UCI form, e.g. ``e2e4`` or ``e7e8q``.

    Identical to :func:`foxchess.move.to_uci`; kept here so that callers working
    with notation do not need to import the move module as well. For a form a
    human can read, use :func:`to_long_algebraic` or :func:`san`.

    ``board`` is unused -- a move encodes its own squares -- but the parameter is
    kept so this sits in the same shape as the other notation functions and a
    caller can swap one for another.
    """
    return to_uci(move)


def to_long_algebraic(board: Board, move: int) -> str:
    """Fully explicit long algebraic: ``Ng1-f3``, ``e2-e4``, ``e5xd6``, ``a7-a8=Q``.

    Every move names both squares, so the result is never ambiguous. That is the
    whole point of the form: unlike :func:`san` it needs no disambiguation rules,
    and unlike :func:`algebraic` it is readable. Use it for logs, transcripts and
    anything that must round-trip through :func:`parse_long_algebraic`.
    """
    flag = decode_flag(move)
    if flag == CASTLE_KINGSIDE:
        return "O-O"
    if flag == CASTLE_QUEENSIDE:
        return "O-O-O"
    origin = name_of(decode_from(move))
    target = name_of(decode_to(move))
    piece = board.piece_at(decode_from(move))
    prefix = "" if piece_type_of(piece) is PieceType.PAWN else piece_letter(piece).upper()
    separator = "x" if flag in CAPTURE_FLAGS else "-"
    text = f"{prefix}{origin}{separator}{target}"
    if letter := PROMOTION_LETTER.get(flag):
        text += f"={letter}"
    return text


# ---------------------------------------------------------------------------
# SAN input
# ---------------------------------------------------------------------------


def parse_san(board: Board, text: str) -> int:
    """Resolve a SAN move to a packed move in ``board``.

    Forgiving by design: ``e4``, ``Nf3``, ``Nbd2``, ``0-0`` and ``O-O`` are all
    accepted, and a disambiguator may be supplied even when SAN would omit it.
    """
    cleaned = text.strip().rstrip("+#!?").replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    if not cleaned:
        raise NotationError("empty move text")

    if cleaned in ("O-O", "O-O-O"):
        return _parse_castle(board, cleaned == "O-O")

    match = _SAN_BODY.match(cleaned)
    if match is None:
        raise NotationError(f"cannot parse SAN move: {text!r}")

    target = square_from_name(match["target"])
    piece_type = _PIECE_CHARS[match["piece"]] if match["piece"] else PieceType.PAWN

    # SAN writes the promotion piece in upper case ("a8=Q"); UCI writes it in lower
    # case. Accept either so that the two forms are interchangeable.
    promotion_text = (match["promotion"] or "").lstrip("=").lower()
    promotion = _UCI_PROMOTION.get(promotion_text) if promotion_text else None
    if promotion_text and promotion is None:
        raise NotationError(f"invalid promotion piece in {text!r}")

    # A disambiguator is optional even where SAN would insist on it, so a bare
    # ``d5`` resolves as long as exactly one piece can reach d5.
    hint_file, hint_rank = match["file"], match["rank"]
    wanted_file = "abcdefgh".index(hint_file) if hint_file else None
    wanted_rank = int(hint_rank) - 1 if hint_rank else None

    candidates: list[int] = []
    for move in board.legal_moves():
        if decode_to(move) != target:
            continue
        if piece_type_of(board.piece_at(decode_from(move))) is not piece_type:
            continue
        origin = decode_from(move)
        if wanted_file is not None and file_of(origin) != wanted_file:
            continue
        if wanted_rank is not None and rank_of(origin) != wanted_rank:
            continue
        if promotion is not None and PROMOTION_PIECE.get(decode_flag(move)) is not promotion:
            continue
        candidates.append(move)

    if not candidates:
        raise NotationError(f"{text!r} is not legal in {board.fen()}")
    if len(candidates) == 1:
        return candidates[0]

    # A pawn push never lands on an occupied square, so that filter resolves the
    # ``e4`` vs ``exd5`` style collisions without guessing.
    survivors = [m for m in candidates if board.piece_at(decode_to(m)) == EMPTY]
    if piece_type is PieceType.PAWN and len(survivors) == 1:
        return survivors[0]
    raise NotationError(f"{text!r} is ambiguous: {[to_uci(c) for c in candidates]}")


def _parse_castle(board: Board, kingside: bool) -> int:
    color = int(board.side)
    king_from = 4 if color == 0 else 60
    king_to = (6 if kingside else 2) + (0 if color == 0 else 56)
    candidate = encode_int(king_from, king_to, CASTLE_KINGSIDE if kingside else CASTLE_QUEENSIDE)
    if candidate in board.legal_moves():
        return candidate
    raise NotationError(f"castling is not legal in {board.fen()}")


_EXPLICIT_LONG: Final = re.compile(
    r"(?P<piece>[KQRBN])?(?P<origin>[a-h][1-8])(?P<target>[a-h][1-8])(?P<promotion>[QRBN])?"
)


def _explicit_to_uci(text: str) -> str | None:
    """Normalise fully explicit long algebraic to UCI, or return None.

    ``Ng1-f3``, ``e5xd6`` and ``a7xb8=Q`` all collapse to UCI once the piece
    letter, the separators and the promotion marker are peeled off. Both squares
    must be present: looser forms such as ``xd5`` are SAN's business, not ours.
    """
    compact = re.sub(r"[x:\-=]", "", text).replace(" ", "").rstrip(".")
    if not (match := _EXPLICIT_LONG.fullmatch(compact)):
        return None
    promotion = match["promotion"] or ""
    return f"{match['origin']}{match['target']}{promotion.lower()}"


def parse_long_algebraic(board: Board, text: str) -> int:
    """Accept ``Ng1-f3``, ``e2-e4``, ``e5xd6``, ``a7-a8=Q``, and looser SAN.

    Anything the explicit form does not cover falls back to UCI and then to SAN,
    so ``e4``, ``e2e4``, ``Nf3``, ``e8Q`` and ``xd5`` all resolve too.
    """
    cleaned = text.strip()
    if explicit := _explicit_to_uci(cleaned):
        try:
            return from_uci(explicit, board.legal_moves())
        except ValueError:
            pass
    try:
        return from_uci(cleaned, board.legal_moves())
    except ValueError:
        return parse_san(board, cleaned)


# ---------------------------------------------------------------------------
# Games and PGN
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Game:
    """A game: headers plus a move list in SAN."""

    headers: dict[str, str] = field(default_factory=dict)
    san_moves: list[str] = field(default_factory=list)
    start_fen: str | None = None
    result: str = "*"

    def board(self) -> Board:
        return Board(self.start_fen) if self.start_fen else Board()

    def start_move_number(self) -> int:
        if self.start_fen:
            return Board(self.start_fen).fullmove_number
        return 1

    def start_color(self) -> Color:
        return self.board().side

    def side_at(self, index: int) -> Color:
        """Whose move it is at ``index`` (0-based) in the move list."""
        start = self.start_color()
        return start if index % 2 == 0 else start.opposite

    def replay(self) -> Board:
        """Rebuild the final position by replaying every move."""
        board = self.board()
        for text in self.san_moves:
            board.make_move(parse_san(board, text))
        return board

    def ply(self) -> int:
        return len(self.san_moves)

    def numbered_moves(self) -> Iterator[tuple[int | None, str]]:
        """``(move_number, san)`` pairs; ``None`` for Black's moves."""
        number = self.start_move_number()
        for index, text in enumerate(self.san_moves):
            is_white = self.side_at(index) is Color.WHITE
            yield (number if is_white else None), text
            if not is_white:
                number += 1


SEVEN_TAG_ROSTER: Final[tuple[str, ...]] = (
    "Event", "Site", "Date", "Round", "White", "Black", "Result",
)


def to_pgn(game: Game, *, width: int = 80) -> str:
    """Serialise a :class:`Game` as PGN, with a seven-tag roster and wrapped movetext."""
    lines = [
        f'[{key} "{game.headers.get(key, "?" if key != "Result" else game.result)}"]'
        for key in SEVEN_TAG_ROSTER
    ]
    lines.extend(
        f'[{key} "{value}"]' for key, value in sorted(game.headers.items())
        if key not in SEVEN_TAG_ROSTER
    )
    # A game that did not start from the standard position is only replayable if
    # the position itself is written down. Without this tag a non-standard PGN
    # parses back into a game that starts from the opening position, and replaying
    # it either fails outright or, worse, silently produces a different game.
    if game.start_fen and game.start_fen != STARTING_FEN:
        lines.append('[SetUp "1"]')
        lines.append(f'[FEN "{game.start_fen}"]')

    tokens: list[str] = []
    for number, text in game.numbered_moves():
        if number is not None:
            tokens.append(f"{number}.")
        tokens.append(text)
    tokens.append(game.result)

    body: list[str] = []
    line = ""
    for token in tokens:
        candidate = f"{line} {token}".strip()
        if line and len(candidate) > width:
            body.append(line)
            line = token
        else:
            line = candidate
    if line:
        body.append(line)

    return "\n".join([*lines, "", *body]) + "\n"


_TAG: Final = re.compile(r'\[\s*(\w+)\s+"([^"]*)"\s*\]')


def parse_pgn(text: str) -> Game:
    """Parse a single-game PGN.

    Handles the seven-tag roster plus any extra headers, ``{}`` comments, ``;``
    line comments, ``$n`` annotation glyphs, move numbers and results.
    Variations in parentheses are skipped rather than parsed — foxchess plays
    linear games, so round-tripping them would only invite bugs.
    """
    game = Game()
    for key, value in _TAG.findall(text):
        game.headers[key] = value

    start = text.find("[")
    movetext = text
    if start != -1:
        end = text.rfind("]")
        if end > start:
            movetext = text[end + 1 :]

    moves, result = _tokenise(movetext)
    game.san_moves = moves
    # The tag pair wins over the movetext token, and falls back to ``*``.
    game.result = game.headers.get("Result", result)

    setup = game.headers.get("FEN")
    if setup:
        game.start_fen = setup
    return game


def _tokenise(movetext: str) -> tuple[list[str], str]:
    """Split movetext into move tokens and a result.

    Comments (``{}`` and ``;``), variations in parentheses, ``$n`` annotation
    glyphs, move numbers and trailing ``!``/``?`` are discarded. Variations are
    skipped rather than parsed: foxchess plays linear games, so round-tripping
    them would only add ways to be wrong.
    """
    out: list[str] = []
    token = ""
    depth = 0
    index = 0
    length = len(movetext)

    def flush() -> None:
        nonlocal token
        if token:
            out.append(token)
            token = ""

    while index < length:
        char = movetext[index]
        if char == "{":
            while index < length and movetext[index] != "}":
                index += 1
            index += 1
            continue
        if char == ";":
            while index < length and movetext[index] != "\n":
                index += 1
            continue
        if char == "$":
            index += 1
            while index < length and movetext[index].isdigit():
                index += 1
            continue
        if char == "(":
            depth += 1
            flush()
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth:
            index += 1
            continue
        if char.isspace():
            flush()
            index += 1
            continue
        token += char
        index += 1

    flush()

    # Strip the annotation glyphs some GUIs append to a move, e.g. "e4?!".
    cleaned: list[str] = []
    result = "*"
    for item in out:
        # Not `item = item[:-1]`: rebinding the loop variable would make the
        # annotation stripped here look like a different token to the reader.
        token = item
        while token and token[-1] in "!?…":
            token = token[:-1]
        if not token:
            continue
        if token in RESULTS:
            result = token
        elif not (token[0].isdigit() or token[0] in "!?…"):
            cleaned.append(token)
    return cleaned, result


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------


class RepetitionTracker:
    """Counts occurrences of the current position within a recent window.

    The window is bounded at 200 plies: the fifty-move rule means a repetition
    older than 100 plies can no longer be claimed, so tracking further back
    wastes memory and time on information that cannot change the result.
    """

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self.keys: list[int] = []

    def record(self, board: Board) -> None:
        self.keys.append(board.key)
        if len(self.keys) > self.window:
            del self.keys[: len(self.keys) - self.window]

    def undo(self) -> None:
        if self.keys:
            self.keys.pop()

    def count(self) -> int:
        """How many times the current position has occurred in the window."""
        if not self.keys:
            return 0
        current = self.keys[-1]
        return sum(1 for key in self.keys if key == current)

    def is_threefold(self) -> bool:
        return self.count() >= 3

    def is_fivefold(self) -> bool:
        return self.count() >= 5

    def reset(self) -> None:
        self.keys.clear()


def file_rank(sq: int) -> str:
    """``'e4'`` for a square index."""
    return name_of(sq)


def square_index(name: str) -> int:
    return square_from_name(name)


def side_name(color: Color) -> str:
    return "white" if color is Color.WHITE else "black"
