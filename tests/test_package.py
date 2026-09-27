"""Tests for the package surface itself.

The re-exports in ``__init__`` are the first thing a user touches and the last
thing anyone tests, so a name can quietly stop existing and break an import
somewhere else entirely. These keep the public surface honest.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import foxchess


def test_every_advertised_name_actually_exists() -> None:
    missing = [name for name in foxchess.__all__ if not hasattr(foxchess, name)]
    assert not missing, f"__all__ advertises names that do not exist: {missing}"


def test_the_top_level_functions_shadow_their_own_modules() -> None:
    """`foxchess.search` is the function; the module needs `importlib`.

    Pinned because it has surprised twice: the package re-exports functions
    under the same names as the submodules that define them, so ``from
    foxchess import search`` yields something you cannot call ``Searcher`` on.
    Both are legitimate -- the flat top level is the friendly API -- but a
    caller reaching for the module has to know to ask for it explicitly.
    """
    for name in ("perft", "search", "divide", "san", "to_pgn"):
        assert callable(getattr(foxchess, name)), name
        module = importlib.import_module(f"foxchess.{_SUBMODULE_OF[name]}")
        assert getattr(module, name) is getattr(foxchess, name)


_SUBMODULE_OF = {
    "perft": "perft",
    "search": "search",
    "divide": "perft",
    "san": "notation",
    "to_pgn": "notation",
}


def test_a_module_with_no_shadowing_name_is_still_reachable_by_attribute() -> None:
    """The shadowing is a side effect, not a blanket ban on submodule access."""
    for name in ("notation", "board", "types", "tt", "evaluate", "zobrist"):
        assert inspect.ismodule(getattr(foxchess, name)), name


def test_the_entry_point_modules_stay_unimported_until_asked_for() -> None:
    """`import foxchess` must not drag in the HTTP server or the console scripts.

    The serving and protocol modules are only needed by someone who is actually
    serving or speaking UCI, and importing them costs a socket module and a
    thread module for every plain chess caller. So they stay off the package
    attribute until someone asks.

    Checked in a fresh interpreter: another test in this session will already
    have imported them, and the attribute they leave behind is permanent.
    """
    probe = (
        "import sys, foxchess; "
        "print(sorted(m for m in ('foxchess.api', 'foxchess.uci', 'foxchess.cli')"
        " if m in sys.modules))"
    )
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    # pytest puts src/ on the path for this process; the child has to be told,
    # or it fails on an import that the parent manages only by accident.
    if (root / "src").is_dir():
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout

    # Asking for one does work, and does load it.
    for name in ("api", "uci", "cli"):
        assert importlib.import_module(f"foxchess.{name}") is not None


def test_all_has_no_duplicates_and_names_the_version() -> None:
    duplicates = [name for name in foxchess.__all__ if foxchess.__all__.count(name) > 1]
    assert not duplicates, duplicates
    assert "__version__" in foxchess.__all__


def test_the_version_is_importable_and_not_a_placeholder() -> None:
    assert foxchess.__version__.count(".") == 2
    assert not foxchess.__version__.endswith(".dev0")


def test_the_version_matches_the_packaging_metadata() -> None:
    """Two version strings drift the moment one is edited alone."""
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():  # installed without the source tree
        pytest.skip("running against an installed package, not the source tree")
    text = pyproject.read_text(encoding="utf-8")
    assert f'version = "{foxchess.__version__}"' in text


def test_every_submodule_imports_cleanly() -> None:
    """A submodule that only fails on import is invisible until someone needs it."""
    for info in pkgutil.iter_modules(foxchess.__path__, f"{foxchess.__name__}."):
        importlib.import_module(info.name)


def test_the_package_ships_a_py_typed_marker() -> None:
    """The classifier claims inline types; without the marker mypy ignores them."""
    marker = importlib.resources.files(foxchess).joinpath("py.typed")
    assert marker.is_file(), "py.typed is missing, so the type hints are decorative"


def test_the_readme_example_works() -> None:
    """The first thing in the README. If this breaks, the README is lying."""
    board = foxchess.Board()
    result = foxchess.search(board, foxchess.SearchLimits(depth=2))
    assert result.best_move != foxchess.NO_MOVE
    assert result.pv_uci()[0] == foxchess.to_uci(result.best_move)
    assert board.fen() == foxchess.Board().fen(), "searching must not disturb the board"


def test_the_one_shot_search_takes_a_fen_as_well_as_a_board() -> None:
    fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    result = foxchess.search(fen, foxchess.SearchLimits(depth=2))
    assert result.best_move != foxchess.NO_MOVE
    assert foxchess.Board(fen).fen() == fen


def test_square_names_round_trip() -> None:
    for name in ("a1", "e4", "d5", "h8"):
        square = foxchess.square_from_name(name)
        assert foxchess.name_of(square) == name


def test_colour_constants_are_members_not_loose_ints() -> None:
    """They moved from module constants to enum members once; keep it that way."""
    assert foxchess.Color.WHITE is not foxchess.Color.BLACK
    assert foxchess.Color.WHITE.opposite is foxchess.Color.BLACK


def test_the_search_sentinel_is_shared_across_modules() -> None:
    """A caller comparing a result against its own NO_MOVE must get the same int."""
    # Reached through importlib because the package re-exports a *function* as
    # `foxchess.search`. Every import spelling short of this one -- `from
    # foxchess import search`, and even `import foxchess.search as ...`, which
    # prefers the attribute once the package is initialised -- hands back that
    # function instead of the module.
    move_module = importlib.import_module("foxchess.move")
    search_module = importlib.import_module("foxchess.search")

    assert move_module.NO_MOVE is search_module.NO_MOVE
    assert search_module.NO_MOVE == foxchess.NO_MOVE
