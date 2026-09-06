"""Static lookup tables bundled with the CARE package.

Data a tool needs at call time and that must not require a network round-trip
or a copy of someone else's dataset on the user's machine. Files here are
resolved via :mod:`importlib.resources`, the same way
:mod:`care.runtime.locales` handles the UI catalogs, so editable installs and
wheels both keep them reachable.

Contents:

* ``arc_agi_1_index.json`` / ``arc_agi_2_index.json`` — the ARC tasks whose
  ``puzzle_id`` :func:`care.runtime.arc_index.lookup_puzzle_id` can recognise
  from demonstrations. One file per dataset, because ARC-AGI-1 and ARC-AGI-2
  are served by separate checkpoints with separate puzzle embeddings. Written
  by hand as::

      [
        {"puzzle_id": "007bbfb7",
         "few_shot": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}]},
        ...
      ]

  ``{"tasks": [...]}`` around that list, and a flat
  ``{"<puzzle_id>": [...pairs...]}`` mapping, are read too. Grids are ARC's own
  lists of rows of colour digits 0-9, and a task's entry must carry ALL of its
  demonstrations — a lookup matches the complete set.

  Both files are optional: without one, its solver still works when given a
  ``puzzle_id`` directly, and reports that a task is unrecognised when given
  only demonstrations.
"""
