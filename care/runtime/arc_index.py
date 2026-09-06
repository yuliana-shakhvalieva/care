"""Recognise an ARC-AGI task from its demonstration pairs.

STARM's ARC checkpoint keeps everything it knows about a task in a learned
puzzle embedding, selected by ``puzzle_id`` — the task's ARC-AGI id, which is
the stem of its JSON file. A planner handed an ARC puzzle has the
demonstrations but no idea of that id, so ``solve_arc_agi_1`` /
``solve_arc_agi_2`` would be unusable without a way back from one to the other.

This module is that way back. ``care/runtime/data/arc_agi_{1,2}_index.json``
ship inside the package and list, per task, its id and its demonstrations::

    [{"puzzle_id": "007bbfb7", "few_shot": [{"input": [[..]], "output": [[..]]}]}]

so the files stay readable and hand-editable; the hashing that makes lookups
fast happens here, once, on load. No ARC dataset has to be present on the
machine and nothing is fetched at call time. There is one file per dataset
because ARC-AGI-1 and ARC-AGI-2 are served by separate checkpoints, each with
its own puzzle embeddings.

One task, one entry: the entry holds the hash of each of that task's
demonstrations. A lookup hashes what it was given and compares the set with an
entry's, so demonstrations may come in any order — but all of them are needed,
since a subset would name a task on partial evidence.

The table is read once per process and memoised. It is read-only, so the
"tools must be stateless" rule in :mod:`care.builtin_tools` (a tool callable is
shared across parallel steps) is not in play.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from importlib.resources import files
from typing import Any

_log = logging.getLogger("care.arc_index")

#: The hand-written files this module reads, inside ``care/runtime/data/``.
#: One per ARC dataset, because each is served by its own checkpoint with its
#: own puzzle embeddings — an id from one is meaningless to the other.
ARC_AGI_1_INDEX = "arc_agi_1_index.json"
ARC_AGI_2_INDEX = "arc_agi_2_index.json"


def pair_hash(pair: Any) -> str:
    """Hash one ``{"input": grid, "output": grid}`` demonstration pair.

    Grids are normalised to lists of lists of ints, so a tuple, a
    numpy-derived list and a JSON round-trip all agree — the file is hashed on
    load and the call arguments at call time, and the two must match.

    Returns an empty string when the pair isn't shaped like a demonstration.
    """
    if not isinstance(pair, dict):
        return ""
    try:
        canonical = {
            "input": _normalise_grid(pair.get("input")),
            "output": _normalise_grid(pair.get("output")),
        }
    except (TypeError, ValueError):
        return ""
    if not canonical["input"] or not canonical["output"]:
        return ""
    blob = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalise_grid(grid: Any) -> list[list[int]]:
    """A grid as plain nested ints, or ``[]`` when it isn't one."""
    if not isinstance(grid, (list, tuple)) or not grid:
        return []
    rows: list[list[int]] = []
    for row in grid:
        if not isinstance(row, (list, tuple)) or not row:
            return []
        rows.append([int(cell) for cell in row])
    return rows


def _build_table(payload: Any) -> dict[str, frozenset[str]]:
    """Turn the shipped records into a ``puzzle_id -> demonstration hashes`` lookup.

    The file is written by hand, so it stays in the shape a person can read
    and check — the demonstrations as they appear in ARC, next to the id they
    belong to. Hashing is ours: it happens once here, on load.

    Accepted shapes, in order of preference:

    * a list of ``{"puzzle_id": ..., "few_shot": [...]}`` records;
    * ``{"tasks": [ ...same records... ]}``;
    * a flat ``{"<puzzle_id>": [ ...pairs... ]}`` mapping.

    Malformed records are skipped with a warning rather than failing the
    load: one bad entry should not take the whole lookup down.
    """
    records: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            payload = tasks
        else:
            # Flat {puzzle_id: pairs} mapping.
            records = [(str(k), v) for k, v in payload.items()]
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            # `few-shot` as well as `few_shot`: the hyphen is the natural
            # spelling in prose and an easy thing to type into the file.
            pairs = entry.get("few_shot", entry.get("few-shot"))
            records.append((str(entry.get("puzzle_id") or ""), pairs))

    table: dict[str, frozenset[str]] = {}
    for puzzle_id, pairs in records:
        digests = (
            frozenset(d for pair in pairs if (d := pair_hash(pair)))
            if isinstance(pairs, (list, tuple))
            else frozenset()
        )
        if not puzzle_id or not digests:
            _log.warning("ARC index: skipping entry %r (no id or no pairs)", puzzle_id)
            continue
        table[puzzle_id] = digests
    return table


@lru_cache(maxsize=4)
def _index(filename: str) -> dict[str, frozenset[str]]:
    """One bundled lookup, empty when its file isn't shipped.

    A missing file is a normal state, not a failure: the package can be built
    before an index is written, and callers degrade to "task not recognised"
    rather than crashing a chain step.
    """
    try:
        raw = (
            files("care.runtime.data")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        _log.info("no bundled %s; few-shot lookup is unavailable", filename)
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        _log.warning("%s is not valid JSON: %s", filename, exc)
        return {}
    table = _build_table(payload)
    if not table:
        _log.warning("%s yielded no usable tasks", filename)
    return table


def lookup_puzzle_id(pairs: Any, filename: str) -> str:
    """The ARC-AGI task id these demonstrations belong to, or ``""``.

    ``filename`` picks which dataset's index to search — the ARC-AGI-1 and
    ARC-AGI-2 checkpoints have separate embedding tables, so an id resolved
    against the wrong one would select the wrong task.

    ``pairs`` is the task's demonstrations — a sequence of
    ``{"input": grid, "output": grid}`` dicts. Each is hashed and the set is
    compared with the task's own: order does not matter, but ALL of the
    task's demonstrations have to be there. A missing one, or an extra one
    the task does not have, means no match.

    ``""`` means no match: the task isn't in the file, its demonstrations
    were not all passed, or the grids differ from the ones recorded.
    """
    if not isinstance(pairs, (list, tuple)):
        return ""
    given = {d for pair in pairs if (d := pair_hash(pair))}
    if not given:
        return ""
    for puzzle_id, digests in _index(filename).items():
        if given == digests:
            return puzzle_id
    return ""


__all__ = [
    "ARC_AGI_1_INDEX",
    "ARC_AGI_2_INDEX",
    "lookup_puzzle_id",
    "pair_hash",
]
