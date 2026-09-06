"""Tests for the bundled standard tools and their wiring.

Covers:
* the pure tools (``calculator`` / ``current_datetime`` / HTML→text),
* ``web_search`` formatting + graceful no-key / provider-error paths,
* ``register_builtin_tools`` activation onto a stub context,
* ``builtin_tool_specs`` discovery shape, and
* ``executor._apply_default_tools`` registering builtins from config.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import httpx
import pytest
import respx

from care import builtin_tools
from care.config import CareConfig


class _StubCtx:
    """Minimal stand-in for CARL's ReasoningContext tool registry."""

    def __init__(self) -> None:
        self.registered: dict[str, object] = {}
        self.tags: dict[str, set[str]] = {}

    def register_tool(self, name, fn, *, timeout=None, tags=None):  # noqa: ANN001
        self.registered[name] = fn
        self.tags[name] = set(tags or [])


# ---------------------------------------------------------------------------
# Pure tools
# ---------------------------------------------------------------------------


def test_calculator_evaluates():
    assert builtin_tools.calculator("2 * (3 + 4)") == "14"


def test_calculator_empty_is_friendly():
    assert "empty expression" in builtin_tools.calculator("   ")


def test_calculator_error_does_not_raise():
    out = builtin_tools.calculator("1 / 0")
    assert "calculator error" in out


def test_current_datetime_is_iso_utc():
    out = builtin_tools.current_datetime()
    assert "T" in out and "+00:00" in out and "(" in out  # ISO + weekday/long-date


def test_html_to_text_strips_tags_and_scripts():
    html = (
        "<html><head><style>.x{}</style></head>"
        "<body><script>evil()</script><p>Hello&nbsp;<b>world</b></p></body></html>"
    )
    text = builtin_tools._html_to_text(html)
    assert "Hello" in text and "world" in text
    assert "evil" not in text and "<p>" not in text


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def test_bias_recency():
    from datetime import datetime, timezone

    yr = datetime.now(timezone.utc).year
    # recency intent (EN + RU) → current year appended
    assert str(yr) in builtin_tools._bias_recency("latest Imagine Dragons track")
    assert str(yr) in builtin_tools._bias_recency("последний трек Imagine Dragons")
    # a stale year gets rewritten to the current one (not duplicated)
    assert builtin_tools._bias_recency("Imagine Dragons latest 2021") == f"Imagine Dragons latest {yr}"
    # non-temporal query left untouched
    assert builtin_tools._bias_recency("history of the Roman empire") == "history of the Roman empire"


def test_web_search_no_key_falls_back_to_duckduckgo(monkeypatch):
    """No API key → keyless DuckDuckGo fallback so search works out of the
    box (previously returned a 'not configured' line)."""
    seen: dict[str, str] = {}

    async def fake_search(provider, api_key, query, max_results):  # noqa: ANN001
        seen["provider"] = provider
        return None, [{"title": "DDG", "url": "https://d.dg", "content": "ok"}]

    monkeypatch.setattr(builtin_tools, "_search", fake_search)
    ws = builtin_tools._make_web_search("tavily", None, 5)
    out = asyncio.run(ws("weather in Moscow"))
    assert seen["provider"] == "duckduckgo"  # downgraded to the keyless engine
    assert "[1] DDG" in out


def test_web_search_empty_query():
    ws = builtin_tools._make_web_search("tavily", "key", 5)
    assert "empty query" in asyncio.run(ws("  "))


def test_web_search_formats_results(monkeypatch):
    async def fake_search(provider, api_key, query, max_results):  # noqa: ANN001
        assert provider == "tavily" and api_key == "key"
        return ("Moscow is +18°C and clear.", [
            {"title": "Moscow weather", "url": "https://ex.com", "content": "+18°C"},
            {"title": "Forecast", "url": "https://ex.org", "content": "rain"},
        ])

    monkeypatch.setattr(builtin_tools, "_search", fake_search)
    ws = builtin_tools._make_web_search("tavily", "key", 5)
    out = asyncio.run(ws("Moscow weather"))
    assert "Answer: Moscow is +18°C and clear." in out  # provider answer leads
    assert "[1] Moscow weather" in out
    assert "https://ex.com" in out
    assert "[2] Forecast" in out


def test_web_search_provider_error_is_graceful(monkeypatch):
    async def boom(*a, **k):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(builtin_tools, "_search", boom)
    ws = builtin_tools._make_web_search("tavily", "key", 5)
    out = asyncio.run(ws("x"))
    assert "web_search error" in out and "429" in out


def test_web_search_dedupes_urls(monkeypatch):
    """Repeated URLs (e.g. across fallback engines) collapse to one entry."""

    async def fake(provider, api_key, query, max_results):  # noqa: ANN001
        return None, [
            {"title": "A", "url": "https://x", "content": "1"},
            {"title": "B", "url": "https://x", "content": "2"},  # dup URL
            {"title": "C", "url": "https://y", "content": "3"},
        ]

    monkeypatch.setattr(builtin_tools, "_search", fake)
    ws = builtin_tools._make_web_search("tavily", "k", 5)
    out = asyncio.run(ws("q"))
    assert "[1] A" in out and "[2] C" in out and "[3]" not in out


async def _no_sleep(_seconds):  # noqa: ANN001 — injected backoff stub
    return None


def test_search_resilient_retries_transient_then_succeeds(monkeypatch):
    """A transient 429 is retried (with injected no-op backoff) and the
    second attempt's result is returned."""
    calls = {"n": 0}

    async def flaky(provider, api_key, query, max_results):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("POST", "https://x"),
                response=httpx.Response(429),
            )
        return "ans", [{"title": "T", "url": "u", "content": "c"}]

    monkeypatch.setattr(builtin_tools, "_search", flaky)
    answer, results = asyncio.run(
        builtin_tools._search_resilient("tavily", "k", "q", 3, sleep=_no_sleep)
    )
    assert calls["n"] == 2 and answer == "ans" and results


def test_search_resilient_falls_back_to_duckduckgo(monkeypatch):
    """A non-transient primary failure falls through to keyless DuckDuckGo."""

    async def fake(provider, api_key, query, max_results):  # noqa: ANN001
        if provider == "tavily":
            raise RuntimeError("boom")  # non-transient → next provider
        if provider == "duckduckgo":
            return None, [{"title": "DDG", "url": "https://d", "content": "ok"}]
        return None, []

    monkeypatch.setattr(builtin_tools, "_search", fake)
    answer, results = asyncio.run(
        builtin_tools._search_resilient("tavily", "k", "q", 3, sleep=_no_sleep)
    )
    assert results and results[0]["title"] == "DDG"


@respx.mock
def test_search_tavily_400_downgrades_to_basic():
    """Tavily 400 on advanced search retries once with basic depth."""
    route = respx.post("https://api.tavily.com/search").mock(
        side_effect=[
            httpx.Response(400, json={"detail": "bad request"}),
            httpx.Response(
                200,
                json={
                    "answer": "ok",
                    "results": [{"title": "T", "url": "u", "content": "c"}],
                },
            ),
        ]
    )
    answer, results = asyncio.run(builtin_tools._search("tavily", "k", "q", 3))
    assert answer == "ok" and results[0]["url"] == "u"
    assert route.call_count == 2  # advanced 400 → basic retry


@respx.mock
def test_search_serper_maps_results():
    respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "answerBox": {"answer": "42"},
                "organic": [{"title": "T", "link": "https://x", "snippet": "s"}],
            },
        )
    )
    answer, results = asyncio.run(builtin_tools._search("serper", "k", "q", 3))
    assert answer == "42"
    assert results[0]["url"] == "https://x" and results[0]["content"] == "s"


@respx.mock
def test_search_exa_maps_results():
    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://e", "text": "body"}]},
        )
    )
    answer, results = asyncio.run(builtin_tools._search("exa", "k", "q", 3))
    assert answer is None
    assert results[0]["url"] == "https://e" and results[0]["content"] == "body"


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------


def test_fetch_url_empty():
    fu = builtin_tools._make_fetch_url(4000)
    assert "empty url" in asyncio.run(fu(""))


# ---------------------------------------------------------------------------
# http_request
# ---------------------------------------------------------------------------


def test_http_request_empty_url():
    hr = builtin_tools._make_http_request(4000)
    assert "empty url" in asyncio.run(hr(""))


def test_coerce_mapping():
    assert builtin_tools._coerce_mapping('{"a": 1}') == {"a": 1}
    assert builtin_tools._coerce_mapping({"b": 2}) == {"b": 2}
    assert builtin_tools._coerce_mapping("not json") is None
    assert builtin_tools._coerce_mapping(None) is None


def test_http_request_success(monkeypatch):
    import httpx

    class _Resp:
        status_code = 201
        text = '{"ok": true}'
        headers = {"content-type": "application/json"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, verb, url, **kw):  # noqa: ANN001
            assert verb == "POST" and url == "https://api.example.com/x"
            assert kw.get("json") == {"q": 1}
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    hr = builtin_tools._make_http_request(4000)
    out = asyncio.run(hr("api.example.com/x", method="post", json_body='{"q": 1}'))
    assert "HTTP 201 POST" in out
    assert "https://api.example.com/x" in out
    assert '{"ok": true}' in out


# ---------------------------------------------------------------------------
# STARM solvers
# ---------------------------------------------------------------------------


class _StarmCfg:
    """Duck-typed ToolsConfig with three STARM servers configured."""

    starm_host = "http://localhost"
    starm_ports = {"sudoku": 8081, "game_of_life": 8082, "arc_agi_1": 8083}
    starm_timeout = 5.0
    enable_code_exec = False


def _solvers(cfg=None):
    ctx = _StubCtx()
    builtin_tools.register_builtin_tools(ctx, cfg or _StarmCfg(), CareConfig().sandbox)
    return ctx


def test_starm_registers_one_solver_per_configured_task():
    ctx = _solvers()
    assert {"solve_sudoku", "solve_game_of_life", "solve_arc_agi_1"} <= set(ctx.registered)
    assert "starm_solve" not in ctx.registered  # no generic entry point
    assert "algorithmic" in ctx.tags["solve_sudoku"]


def test_starm_registers_nothing_without_configured_servers():
    """A solver with no server behind it would only mislead the planner."""
    ctx = _solvers(CareConfig().tools)
    assert not [name for name in ctx.registered if name.startswith("solve_")]


def test_starm_solver_signatures():
    ctx = _solvers()
    assert list(inspect.signature(ctx.registered["solve_sudoku"]).parameters) == [
        "problem"
    ]
    # ARC selects a learned puzzle embedding, so its id is a real argument.
    assert list(inspect.signature(ctx.registered["solve_arc_agi_1"]).parameters) == [
        "input_grid",
        "puzzle_id",
        "few_shot",
    ]
    # The evolution count is an argument, not punctuation in the pattern.
    assert list(inspect.signature(ctx.registered["solve_game_of_life"]).parameters) == [
        "input_pattern",
        "evolutions",
    ]


@respx.mock
def test_starm_solver_posts_to_its_own_port():
    route = respx.post("http://localhost:8081/generate").mock(
        return_value=httpx.Response(
            200, json={"output": "534678912", "steps": 12, "max_steps": 16}
        )
    )
    out = asyncio.run(_solvers().registered["solve_sudoku"]("53..."))
    assert "534678912" in out
    assert "12/16" in out  # the ACT step count is carried through
    assert json.loads(route.calls[0].request.content) == {
        "task": "sudoku",
        "input": "53...",
    }


@respx.mock
def test_starm_solvers_do_not_share_a_port():
    """Each task has its own server; the tool name picks it."""
    respx.post("http://localhost:8082/generate").mock(
        return_value=httpx.Response(
            200, json={"output": "bob$bob$bob", "steps": 2, "max_steps": 8}
        )
    )
    out = asyncio.run(_solvers().registered["solve_game_of_life"]("bbb$ooo$bbb", 1))
    assert "bob$bob$bob" in out


@respx.mock
def test_game_of_life_joins_the_pattern_and_the_step_count():
    """`evolutions` is its own argument because planners forget a '|N'
    suffix; the tool is what builds the string the server parses."""
    route = respx.post("http://localhost:8082/generate").mock(
        return_value=httpx.Response(200, json={"output": "bob$bob$bob"})
    )
    gol = _solvers().registered["solve_game_of_life"]
    asyncio.run(gol("bbb$ooo$bbb", 1))
    assert json.loads(route.calls[0].request.content)["input"] == "bbb$ooo$bbb|1"
    # Planners routinely pass numbers as strings.
    asyncio.run(gol("bbb$ooo$bbb", "3"))
    assert json.loads(route.calls[1].request.content)["input"] == "bbb$ooo$bbb|3"


@respx.mock(assert_all_called=False)
def test_game_of_life_rejects_an_unusable_evolution_count(respx_mock):
    gol = _solvers().registered["solve_game_of_life"]
    assert "whole number" in asyncio.run(gol("bbb$ooo$bbb", "once"))
    assert "cannot be negative" in asyncio.run(gol("bbb$ooo$bbb", -1))
    assert not respx_mock.calls  # neither reaches the server


# --- solve_arc: puzzle_id / few_shot ---------------------------------------

_PAIR_A = {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
_PAIR_B = {"input": [[2, 2], [0, 0]], "output": [[0, 0], [2, 2]]}
_PAIR_C = {"input": [[5]], "output": [[6]]}


#: The shape of the hand-written package data, in miniature.
from care.runtime.arc_index import ARC_AGI_1_INDEX as _ARC1

_ARC_INDEX_FILE = [
    {"puzzle_id": "007bbfb7", "few_shot": [_PAIR_A, _PAIR_B]},
    {"puzzle_id": "beefbeef", "few-shot": [_PAIR_C]},  # `few-shot` is read too
]


@pytest.fixture
def arc_index(monkeypatch):
    """Load the fixture file through the real parser, then serve it for
    whichever dataset index is asked for."""
    from care.runtime import arc_index as module

    table = module._build_table(_ARC_INDEX_FILE)
    monkeypatch.setattr(module, "_index", lambda filename: table)
    return module


def test_arc_index_holds_one_entry_of_hashes_per_task(arc_index):
    """One task, one entry — carrying the hash of each of its
    demonstrations."""
    table = arc_index._build_table(_ARC_INDEX_FILE)
    assert len(table) == len(_ARC_INDEX_FILE)
    assert table["007bbfb7"] == frozenset(
        {arc_index.pair_hash(_PAIR_A), arc_index.pair_hash(_PAIR_B)}
    )


def test_arc_index_reads_the_accepted_file_shapes(arc_index):
    module = arc_index
    plain = module._build_table(_ARC_INDEX_FILE)
    assert module._build_table({"tasks": _ARC_INDEX_FILE}) == plain
    assert module._build_table({"007bbfb7": [_PAIR_A]}) == {
        "007bbfb7": frozenset({module.pair_hash(_PAIR_A)})
    }
    # An entry without an id, or without pairs, is skipped rather than fatal.
    assert module._build_table([{"puzzle_id": "", "few_shot": [_PAIR_B]}]) == {}
    assert module._build_table([{"puzzle_id": "x"}]) == {}


def test_arc_pair_hash_is_stable_across_a_json_round_trip(arc_index):
    """The file is hashed on load, the call arguments at call time — both
    must land on the same digest."""
    assert arc_index.pair_hash(_PAIR_A) == arc_index.pair_hash(
        json.loads(json.dumps(_PAIR_A))
    )
    assert arc_index.pair_hash({"input": "not a grid"}) == ""


def test_arc_lookup_ignores_the_order_of_the_demonstrations(arc_index):
    assert arc_index.lookup_puzzle_id([_PAIR_A, _PAIR_B], _ARC1) == "007bbfb7"
    assert arc_index.lookup_puzzle_id([_PAIR_B, _PAIR_A], _ARC1) == "007bbfb7"
    assert arc_index.lookup_puzzle_id([_PAIR_C], _ARC1) == "beefbeef"


def test_arc_lookup_needs_all_of_the_demonstrations(arc_index):
    """Naming a task from part of its evidence would pick an embedding on a
    guess, so the set has to be complete."""
    assert arc_index.lookup_puzzle_id([_PAIR_A], _ARC1) == ""  # 007bbfb7 has two
    assert arc_index.lookup_puzzle_id([_PAIR_B], _ARC1) == ""


def test_arc_lookup_returns_empty_when_a_pair_does_not_belong(arc_index):
    unknown = {"input": [[7]], "output": [[7]]}
    # An extra demonstration the task does not have rules it out.
    assert arc_index.lookup_puzzle_id([_PAIR_A, _PAIR_B, unknown], _ARC1) == ""
    # Pairs from two different tasks belong to neither.
    assert arc_index.lookup_puzzle_id([_PAIR_A, _PAIR_C], _ARC1) == ""
    assert arc_index.lookup_puzzle_id([unknown], _ARC1) == ""
    assert arc_index.lookup_puzzle_id([], _ARC1) == ""
    assert arc_index.lookup_puzzle_id("not pairs") == ""


def test_parse_arc_few_shot_accepts_the_shapes_planners_send():
    parse = builtin_tools._parse_arc_few_shot
    assert parse(json.dumps([_PAIR_A, _PAIR_B])) is not None      # JSON string
    assert parse([_PAIR_A]) is not None                            # real list
    assert len(parse({"train": [_PAIR_A, _PAIR_B], "test": []})) == 2  # ARC task
    assert parse("not json") is None
    assert parse("") is None


@respx.mock
def test_arc_resolves_the_puzzle_id_from_few_shot(arc_index):
    route = respx.post("http://localhost:8083/generate").mock(
        return_value=httpx.Response(200, json={"output": "0 1"})
    )
    out = asyncio.run(
        _solvers().registered["solve_arc_agi_1"](
            "000<eos>010", few_shot=json.dumps([_PAIR_A, _PAIR_B])
        )
    )
    assert "0 1" in out
    assert json.loads(route.calls[0].request.content)["puzzle_id"] == "007bbfb7"


@respx.mock
def test_arc_prefers_an_explicit_puzzle_id_and_says_so(arc_index):
    """Silently dropping few_shot is how a caller ends up believing the
    demonstrations were used."""
    route = respx.post("http://localhost:8083/generate").mock(
        return_value=httpx.Response(200, json={"output": "0 1"})
    )
    out = asyncio.run(
        _solvers().registered["solve_arc_agi_1"](
            "000<eos>010", puzzle_id="beefbeef", few_shot=json.dumps([_PAIR_A, _PAIR_B])
        )
    )
    assert json.loads(route.calls[0].request.content)["puzzle_id"] == "beefbeef"
    assert "ignoring few_shot" in out
    assert "0 1" in out  # the warning rides along with the answer


@respx.mock(assert_all_called=False)
def test_arc_without_either_argument_costs_no_request(respx_mock, arc_index):
    out = asyncio.run(_solvers().registered["solve_arc_agi_1"]("000<eos>010"))
    assert "either `puzzle_id`" in out
    assert not respx_mock.calls


@respx.mock(assert_all_called=False)
def test_arc_unrecognised_few_shot_costs_no_request(respx_mock, arc_index):
    """No embedding exists for a task outside the training set, so there is
    nothing to ask the server."""
    out = asyncio.run(
        _solvers().registered["solve_arc_agi_1"](
            "000<eos>010", few_shot=json.dumps([{"input": [[7]], "output": [[7]]}])
        )
    )
    assert "no task the" in out
    assert not respx_mock.calls


@respx.mock(assert_all_called=False)
def test_arc_unreadable_few_shot_names_the_expected_shape(respx_mock, arc_index):
    out = asyncio.run(
        _solvers().registered["solve_arc_agi_1"]("000<eos>010", few_shot="not json")
    )
    assert "could not read" in out
    assert not respx_mock.calls


@pytest.mark.parametrize("index", ["ARC_AGI_1_INDEX", "ARC_AGI_2_INDEX"])
def test_bundled_arc_index_is_well_formed(index):
    """Guards the hand-written package data, once it exists."""
    from care.runtime import arc_index as module

    filename = getattr(module, index)

    module._index.cache_clear()
    table = module._index(filename)
    if not table:
        pytest.skip(f"{filename} not written yet")
    for puzzle_id, digests in table.items():
        assert puzzle_id and digests
        for digest in digests:
            assert len(digest) == 64
            assert all(c in "0123456789abcdef" for c in digest)


@respx.mock
def test_arc_forwards_the_puzzle_id():
    route = respx.post("http://localhost:8083/generate").mock(
        return_value=httpx.Response(200, json={"output": "0 1", "steps": 1, "max_steps": 4})
    )
    asyncio.run(_solvers().registered["solve_arc_agi_1"]("0 0", "007bbfb7"))
    assert json.loads(route.calls[0].request.content)["puzzle_id"] == "007bbfb7"


@respx.mock(assert_all_called=False)
def test_starm_solver_empty_problem_costs_no_request(respx_mock):
    out = asyncio.run(_solvers().registered["solve_sudoku"]("   "))
    assert "`input_grid`" in out  # the message names this tool's own argument
    assert not respx_mock.calls


@respx.mock
def test_starm_solver_unreachable_server_is_graceful():
    respx.post("http://localhost:8081/generate").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    out = asyncio.run(_solvers().registered["solve_sudoku"]("53..."))
    assert "solve_sudoku error" in out
    assert "Is a STARM server running" in out


@respx.mock
def test_starm_solver_relays_the_servers_rejection():
    """STARM owns the prompt format, so its 422 text is the answer."""
    respx.post("http://localhost:8081/generate").mock(
        return_value=httpx.Response(422, json={"detail": "81 cells expected, got 12"})
    )
    out = asyncio.run(_solvers().registered["solve_sudoku"]("53"))
    assert "81 cells expected, got 12" in out


@respx.mock
def test_starm_solver_empty_answer_is_reported():
    respx.post("http://localhost:8081/generate").mock(
        return_value=httpx.Response(200, json={"output": "  "})
    )
    out = asyncio.run(_solvers().registered["solve_sudoku"]("x"))
    assert "empty answer" in out


@respx.mock
def test_starm_solver_honours_a_remote_host():
    cfg = _StarmCfg()
    cfg.starm_host = "http://gpu-01"
    respx.post("http://gpu-01:8081/generate").mock(
        return_value=httpx.Response(200, json={"output": "ok", "steps": 1, "max_steps": 2})
    )
    assert "ok" in asyncio.run(_solvers(cfg).registered["solve_sudoku"]("x"))


def test_starm_base_url_composition():
    assert builtin_tools._starm_base_url("http://gpu-01", 8081) == "http://gpu-01:8081"
    assert builtin_tools._starm_base_url("gpu-01", 8080) == "http://gpu-01:8080"
    # A port on the host loses to the task's own.
    assert builtin_tools._starm_base_url("http://localhost:9000", 8081) == (
        "http://localhost:8081"
    )


def test_starm_solver_ports_drops_unusable_entries():
    class _Cfg:
        starm_ports = {
            "sudoku": 8081,
            "chess": 9000,  # not a STARM task
            "maze": "nope",  # not a port
            "Game-Of-Life": 8082,  # spelling folded
        }

    assert builtin_tools.starm_solver_ports(_Cfg()) == {
        "sudoku": 8081,
        "game_of_life": 8082,
    }


def test_starm_ports_parse_from_an_env_style_string():
    cfg = CareConfig(tools={"starm_ports": "sudoku=8080, maze=8081"})
    assert cfg.tools.starm_ports == {"sudoku": 8080, "maze": 8081}


def test_starm_ports_parse_from_json():
    cfg = CareConfig(tools={"starm_ports": '{"arc_agi_1": 8083}'})
    assert cfg.tools.starm_ports == {"arc_agi_1": 8083}


def test_starm_ports_empty_by_default():
    assert CareConfig().tools.starm_ports == {}


def test_starm_specs_are_advertised_per_task():
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs(_StarmCfg())}
    assert {"solve_sudoku", "solve_game_of_life", "solve_arc_agi_1"} <= set(by_name)
    sudoku = by_name["solve_sudoku"]["description"]
    assert sudoku.startswith("solve_sudoku(input_grid: str) -> str.")
    assert "task_id" not in sudoku  # the solver is chosen by tool name
    arc = by_name["solve_arc_agi_1"]["description"]
    assert "puzzle_id" in arc
    # Rows go in separated by '<eos>', and the example shows exactly that.
    assert "'000<eos>010<eos>000'" in arc


def test_every_starm_spec_matches_its_solver():
    """The hand-written specs and the registered callables must agree:
    one spec per task, same name, and a signature the description documents."""

    class _AllTasks:
        starm_host = "http://localhost"
        starm_timeout = 5.0
        enable_code_exec = False
        starm_ports = {
            task: 8000 + i
            for i, task in enumerate(builtin_tools._STARM_TASKS)
        }

    cfg = _AllTasks()
    ctx = _StubCtx()
    builtin_tools.register_builtin_tools(ctx, cfg, CareConfig().sandbox)
    specs = {s["name"]: s for s in builtin_tools.builtin_tool_specs(cfg)}

    for task in builtin_tools._STARM_TASKS:
        name = f"solve_{task_id}"
        assert name in ctx.registered, name
        assert name in specs, name
        params = list(inspect.signature(ctx.registered[name]).parameters)
        head = specs[name]["description"].split(" -> str.")[0]
        assert head.startswith(f"{name}("), name
        for param in params:  # every argument is documented in the signature
            assert param in head, (name, param)
        assert specs[name]["source"] == "care:builtin"
        assert specs[name]["tags"]


def test_starm_specs_never_mention_the_engine():
    """The planner picks a tool by what it does; the backend is CARE's
    business, and naming it only spends context."""
    for spec in builtin_tools.builtin_tool_specs(_StarmCfg()):
        if spec["name"].startswith("solve_"):
            assert "starm" not in spec["description"].lower(), spec["name"]


def test_starm_specs_document_the_problem_format():
    """Each solver states its own encoding — that's what stops a planner
    passing the user's sentence through."""
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs(_StarmCfg())}
    assert "81 cells" in by_name["solve_sudoku"]["description"]
    assert "'#' a wall" in by_name["solve_maze"]["description"]
    arithmetic = by_name["solve_arithmetic"]["description"]
    # Postfix, per the dataset builder's evaluate_rpn — the example must parse
    # that way, unlike the infix one in STARM's own input_help.
    assert "REVERSE POLISH" in arithmetic
    assert "'34?5?=35'" in arithmetic
    # The notation is unfolded step by step: postfix is not a format a
    # planner can be assumed to read off the name alone.
    assert "'34+5*' unfolds as (3+4)=7, then 7*5=35" in arithmetic
    gol = by_name["solve_game_of_life"]["description"]
    assert "'bbb$ooo$bbb'" in gol  # the pattern; the tool adds the count
    assert "`input_pattern`: the starting pattern" in gol
    assert "`evolutions`: after how many evolutions" in gol


def test_every_starm_spec_carries_an_example():
    """A format described in prose still gets guessed at; a worked example
    is what a planner copies."""
    class _AllTasks:
        starm_host = "http://localhost"
        starm_timeout = 5.0
        starm_ports = {
            task: 8000 + i
            for i, task in enumerate(builtin_tools._STARM_TASKS)
        }

    for spec in builtin_tools.builtin_tool_specs(_AllTasks()):
        if spec["name"].startswith("solve_"):
            assert "Example" in spec["description"], spec["name"]


def test_starm_specs_absent_without_configured_servers():
    names = {s["name"] for s in builtin_tools.builtin_tool_specs()}
    assert not [n for n in names if n.startswith("solve_")]


# ---------------------------------------------------------------------------
# run_python (sandboxed)
# ---------------------------------------------------------------------------


def test_run_python_empty():
    rp = builtin_tools._make_run_python(CareConfig().sandbox, 30, 4000)
    assert "empty code" in asyncio.run(rp("   "))


def test_run_python_requires_docker_kind():
    cfg = CareConfig()
    cfg.sandbox.kind = "local"
    rp = builtin_tools._make_run_python(cfg.sandbox, 30, 4000)
    out = asyncio.run(rp("print(1)"))
    assert "Docker" in out and "CARE_SANDBOX__KIND=docker" in out


def test_run_python_missing_docker_cli(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *_: None)
    rp = builtin_tools._make_run_python(CareConfig().sandbox, 30, 4000)
    out = asyncio.run(rp("print(1)")).lower()
    assert "docker" in out and "path" in out


# ---------------------------------------------------------------------------
# Activation + discovery
# ---------------------------------------------------------------------------


_EXPECTED = {
    "web_search",
    "fetch_url",
    "http_request",
    "calculator",
    "current_datetime",
    "run_python",
}


def test_register_builtin_tools_registers_expected_names():
    ctx = _StubCtx()
    names = builtin_tools.register_builtin_tools(ctx, CareConfig().tools)
    assert set(names) == _EXPECTED
    assert set(ctx.registered) == set(names)
    # web tools carry the 'web' tag for tag-restricted steps.
    assert "web" in ctx.tags["web_search"]


def test_register_respects_disable_code_exec():
    cfg = CareConfig()
    cfg.tools.enable_code_exec = False
    ctx = _StubCtx()
    names = builtin_tools.register_builtin_tools(ctx, cfg.tools, cfg.sandbox)
    assert "run_python" not in names
    assert "http_request" in names  # http stays — only code-exec gated


def test_register_builtin_tools_skips_context_without_register_tool():
    # Duck-typed guard: a stripped context shouldn't explode the run.
    assert builtin_tools.register_builtin_tools(object(), CareConfig().tools) == []


def test_builtin_tool_specs_shape():
    specs = builtin_tools.builtin_tool_specs()
    by_name = {s["name"]: s for s in specs}
    assert _EXPECTED <= set(by_name)
    for spec in specs:
        assert spec["source"] == "care:builtin"
        assert spec["description"] and isinstance(spec["tags"], list)
    # The signature is baked into the description so MAGE maps inputs.
    assert "query" in by_name["web_search"]["description"]
    assert "run_python(code" in by_name["run_python"]["description"]


def test_specs_exclude_run_python_when_disabled():
    cfg = CareConfig()
    cfg.tools.enable_code_exec = False
    names = {s["name"] for s in builtin_tools.builtin_tool_specs(cfg.tools)}
    assert "run_python" not in names
    assert "http_request" in names


# ---------------------------------------------------------------------------
# Executor wiring
# ---------------------------------------------------------------------------


def test_apply_default_tools_registers_builtins():
    from care.runtime import executor

    ctx = _StubCtx()
    executor._apply_default_tools(ctx, CareConfig())
    assert {"web_search", "calculator"} <= set(ctx.registered)


def test_apply_default_tools_none_config_is_noop():
    from care.runtime import executor

    ctx = _StubCtx()
    executor._apply_default_tools(ctx, None)
    assert ctx.registered == {}


def test_apply_default_tools_respects_disable_flag():
    from care.runtime import executor

    cfg = CareConfig()
    cfg.tools.enable_builtins = False
    ctx = _StubCtx()
    executor._apply_default_tools(ctx, cfg)
    assert "web_search" not in ctx.registered


def test_disabled_builtins_not_advertised_to_mage(tmp_path):
    from care.capability_priming import build_capabilities_for_generation

    cfg = CareConfig()
    cfg.tools.enable_builtins = False
    # Empty cache dir so previously-synthesised tools don't get advertised.
    cfg.tools.synthesized_tools_path = tmp_path / "empty"
    # No builtins + no cached tools → nothing to prime.
    assert build_capabilities_for_generation(cfg) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
