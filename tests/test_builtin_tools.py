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
# starm_solve
# ---------------------------------------------------------------------------


def _starm(host="http://localhost", ports=None, port=8080, timeout=5.0):
    return builtin_tools._make_starm_solve(host, ports, port, timeout)


def test_starm_base_url_appends_the_call_port():
    # The configured host carries no port; the per-call port picks the server.
    assert (
        builtin_tools._starm_base_url("http://gpu-01", 8081, 8080)
        == "http://gpu-01:8081"
    )


def test_starm_base_url_defaults_and_normalises_scheme():
    assert builtin_tools._starm_base_url("gpu-01", None, 8080) == "http://gpu-01:8080"


def test_starm_base_url_call_port_overrides_one_in_the_host():
    assert (
        builtin_tools._starm_base_url("http://localhost:9000", 8081, 8080)
        == "http://localhost:8081"
    )


def test_starm_base_url_keeps_a_host_port_when_no_call_port():
    assert (
        builtin_tools._starm_base_url("http://localhost:9000", None, 8080)
        == "http://localhost:9000"
    )


def test_coerce_port():
    assert builtin_tools._coerce_port("8081") == 8081
    assert builtin_tools._coerce_port(8081) == 8081
    assert builtin_tools._coerce_port("nope") is None
    assert builtin_tools._coerce_port(0) is None
    assert builtin_tools._coerce_port(70000) is None


def test_starm_solve_rejects_a_bad_port():
    out = asyncio.run(_starm()("53 puzzle", task_id="sudoku", port="eight"))
    assert "not a valid TCP port" in out


@respx.mock
def test_starm_solve_posts_generate_and_returns_the_answer():
    route = respx.post("http://localhost:8081/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "task": "sudoku",
                "input": "53...",
                "output": "534678912",
                "steps": 12,
                "max_steps": 16,
                "puzzle_id": None,
            },
        )
    )
    out = asyncio.run(_starm()("53...", task_id="sudoku", port=8081))
    assert "534678912" in out
    assert "12/16" in out  # the ACT step count is carried through
    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body == {"task": "sudoku", "input": "53..."}  # no puzzle_id for non-ARC


@respx.mock
def test_starm_solve_requires_the_task():
    """`task` is not inferred: a planner that omits it gets told to name one.

    It used to be filled in from the server's /info, which rewarded leaving
    it out — and a planner that hasn't chosen a solver usually hasn't
    encoded the puzzle for one either.
    """
    info = respx.get("http://localhost:8080/info")
    out = asyncio.run(_starm()("##..##"))
    assert "`task_id` is required" in out
    assert info.call_count == 0  # answered before any network call
    # The valid answers are named, so the retry can be right.
    for task in ("sudoku", "maze", "arc", "arithmetic", "game_of_life"):
        assert task in out


def test_starm_solve_missing_task_lists_the_configured_tasks():
    """With servers configured, only those are worth naming."""
    out = asyncio.run(_starm(ports={"sudoku": 8080, "maze": 8081})("x"))
    assert "`task_id` is required" in out
    assert "maze" in out and "sudoku" in out
    assert "arithmetic" not in out  # not served here


@respx.mock
def test_starm_solve_skips_info_when_the_task_is_given():
    info = respx.get("http://localhost:8080/info")
    respx.post("http://localhost:8080/generate").mock(
        return_value=httpx.Response(200, json={"output": "x", "steps": 1, "max_steps": 4})
    )
    asyncio.run(_starm()("53...", task_id="sudoku"))
    assert info.call_count == 0  # one call, not two


@respx.mock
def test_starm_solve_forwards_puzzle_id_for_arc():
    route = respx.post("http://localhost:8080/generate").mock(
        return_value=httpx.Response(200, json={"output": "0 1", "steps": 2, "max_steps": 8})
    )
    asyncio.run(_starm()("0 0\n1 1", task_id="arc", puzzle_id="007bbfb7"))
    assert json.loads(route.calls[0].request.content)["puzzle_id"] == "007bbfb7"


@respx.mock
def test_starm_solve_empty_problem_quotes_the_servers_input_format():
    """The format belongs to the checkpoint — ask, never keep a copy."""
    respx.get("http://localhost:8080/info").mock(
        return_value=httpx.Response(
            200,
            json={"task": "sudoku", "input_format": "81 cells, '.' for blanks"},
        )
    )
    out = asyncio.run(_starm()("   ", task_id="sudoku"))
    assert "empty problem" in out
    assert "81 cells" in out


@respx.mock
def test_starm_solve_422_carries_the_servers_diagnosis_and_format():
    respx.post("http://localhost:8080/generate").mock(
        return_value=httpx.Response(422, json={"detail": "81 cells expected, got 12"})
    )
    respx.get("http://localhost:8080/info").mock(
        return_value=httpx.Response(
            200, json={"task": "sudoku", "input_format": "81 cells, '.' for blanks"}
        )
    )
    out = asyncio.run(_starm()("53...", task_id="sudoku"))
    assert "81 cells expected, got 12" in out  # the server's own message
    assert "81 cells, '.' for blanks" in out  # plus how to fix it


@respx.mock
def test_starm_solve_422_without_info_still_reports_the_rejection():
    respx.post("http://localhost:8080/generate").mock(
        return_value=httpx.Response(422, json={"detail": "wrong task"})
    )
    respx.get("http://localhost:8080/info").mock(side_effect=httpx.ConnectError("down"))
    out = asyncio.run(_starm()("x", task_id="sudoku"))
    assert "wrong task" in out


@respx.mock
def test_starm_solve_unreachable_server_is_graceful():
    respx.post("http://localhost:8080/generate").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    out = asyncio.run(_starm()("53...", task_id="sudoku"))
    assert "starm_solve error" in out
    assert "Is a STARM server running" in out


@respx.mock
def test_starm_solve_empty_output_is_reported():
    respx.post("http://localhost:8080/generate").mock(
        return_value=httpx.Response(200, json={"output": "  ", "steps": 1, "max_steps": 4})
    )
    out = asyncio.run(_starm()("x", task_id="sudoku"))
    assert "empty answer" in out


@respx.mock
def test_starm_solve_honours_the_configured_host():
    respx.post("http://gpu-01:8081/generate").mock(
        return_value=httpx.Response(200, json={"output": "ok", "steps": 1, "max_steps": 2})
    )
    out = asyncio.run(_starm(host="http://gpu-01")("x", task_id="sudoku", port=8081))
    assert "ok" in out


def test_starm_solve_registered_with_config_values():
    """The registered tool is bound to the config, not to hardcoded defaults."""
    cfg = CareConfig()
    cfg.tools.starm_host = "http://gpu-02"
    cfg.tools.starm_port = 9100
    ctx = _StubCtx()
    builtin_tools.register_builtin_tools(ctx, cfg.tools, cfg.sandbox)
    with respx.mock:
        route = respx.post("http://gpu-02:9100/generate").mock(
            return_value=httpx.Response(
                200, json={"output": "bound", "steps": 1, "max_steps": 2}
            )
        )
        out = asyncio.run(ctx.registered["starm_solve"]("x", task_id="sudoku"))
    assert "bound" in out and route.call_count == 1


@respx.mock
def test_starm_solve_takes_the_port_from_the_task_map():
    """The deployment knows the port; the prompt only names the task."""
    route = respx.post("http://localhost:8081/generate").mock(
        return_value=httpx.Response(
            200, json={"output": "SSEE", "steps": 3, "max_steps": 8}
        )
    )
    tool = _starm(ports={"sudoku": 8080, "maze": 8081})
    out = asyncio.run(tool("##..##", task_id="maze"))
    assert "SSEE" in out and route.call_count == 1


@respx.mock
def test_starm_solve_task_lookup_ignores_case_and_dashes():
    respx.post("http://localhost:8082/generate").mock(
        return_value=httpx.Response(200, json={"output": "o", "steps": 1, "max_steps": 2})
    )
    tool = _starm(ports={"game_of_life": 8082})
    assert "o" in asyncio.run(tool("bbo$obb|3", task_id="Game-of-Life"))


def test_starm_solve_unconfigured_task_lists_the_configured_ones():
    tool = _starm(ports={"sudoku": 8080, "maze": 8081})
    out = asyncio.run(tool("x", task_id="chess"))
    assert "not a STARM task id" in out
    assert "maze" in out and "sudoku" in out


def test_starm_solve_rejects_a_description_in_task_id():
    """The failure this rename exists for: `task` read as "what to do"."""
    out = asyncio.run(_starm()("bbb$ooo$bbb|1", task_id="Compute next generation"))
    assert "not a STARM task id" in out
    assert "names WHICH solver to use" in out
    assert "game_of_life" in out  # the answer it should have given


def test_starm_solve_accepts_task_as_a_legacy_alias():
    """Chains generated before the rename keep working."""
    with respx.mock:
        respx.post("http://localhost:8080/generate").mock(
            return_value=httpx.Response(
                200, json={"output": "ok", "steps": 1, "max_steps": 2}
            )
        )
        out = asyncio.run(_starm()("x", task="sudoku"))
    assert "ok" in out


def test_starm_solve_single_configured_server_still_needs_the_task():
    """Even with one server, the task is stated rather than assumed."""
    out = asyncio.run(_starm(ports={"sudoku": 8080})("53..."))
    assert "`task_id` is required" in out


@respx.mock
def test_starm_solve_explicit_port_overrides_the_map():
    respx.post("http://localhost:9999/generate").mock(
        return_value=httpx.Response(200, json={"output": "ok", "steps": 1, "max_steps": 2})
    )
    tool = _starm(ports={"sudoku": 8080})
    assert "ok" in asyncio.run(tool("x", task_id="sudoku", port=9999))


def test_starm_ports_parse_from_an_env_style_string():
    cfg = CareConfig(tools={"starm_ports": "sudoku=8080, maze=8081"})
    assert cfg.tools.starm_ports == {"sudoku": 8080, "maze": 8081}


def test_starm_ports_parse_from_json():
    cfg = CareConfig(tools={"starm_ports": '{"arc": 8083}'})
    assert cfg.tools.starm_ports == {"arc": 8083}


def test_starm_ports_drop_unusable_entries_instead_of_crashing():
    cfg = CareConfig(tools={"starm_ports": "sudoku=8080,maze=nope,arc=99999"})
    assert cfg.tools.starm_ports == {"sudoku": 8080}


def test_starm_ports_empty_by_default():
    assert CareConfig().tools.starm_ports == {}


@pytest.fixture(autouse=True)
def _clear_starm_info_cache():
    """`_starm_input_formats` memoises per process; tests mock different
    servers, so the cache must not leak between them."""
    builtin_tools._starm_input_formats.cache_clear()
    yield
    builtin_tools._starm_input_formats.cache_clear()


@respx.mock
def test_starm_spec_names_the_configured_tasks():
    """MAGE can only pick a task it's told exists."""
    respx.get("http://localhost:8080/info").mock(
        return_value=httpx.Response(200, json={"task": "sudoku", "input_format": ""})
    )
    respx.get("http://localhost:8081/info").mock(
        return_value=httpx.Response(200, json={"task": "maze", "input_format": ""})
    )
    cfg = CareConfig()
    cfg.tools.starm_ports = {"sudoku": 8080, "maze": 8081}
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs(cfg.tools)}
    description = by_name["starm_solve"]["description"]
    assert "maze" in description and "sudoku" in description
    # Nothing configured -> no such claim, and no network call.
    plain = {s["name"]: s for s in builtin_tools.builtin_tool_specs()}
    assert "This deployment serves" not in plain["starm_solve"]["description"]


@respx.mock
def test_starm_spec_quotes_each_tasks_input_format():
    """The encoding lives in the checkpoint, so it's fetched, not guessed.

    Without it a planner passes the user's prose straight through and the
    server rejects it — the failure this covers.
    """
    respx.get("http://localhost:8082/info").mock(
        return_value=httpx.Response(
            200,
            json={
                "task": "game_of_life",
                "input_format": "pattern then the generation count: 'bbo$obb|3'",
            },
        )
    )
    cfg = CareConfig()
    cfg.tools.starm_ports = {"game_of_life": 8082}
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs(cfg.tools)}
    description = by_name["starm_solve"]["description"]
    assert "game_of_life — pattern then the generation count: 'bbo$obb|3'" in description


@respx.mock
def test_starm_spec_survives_an_unreachable_server():
    """Advertising is best-effort: a server that's down still gets named."""
    respx.get("http://localhost:9999/info").mock(
        side_effect=httpx.ConnectError("refused")
    )
    cfg = CareConfig()
    cfg.tools.starm_ports = {"maze": 9999}
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs(cfg.tools)}
    assert "maze" in by_name["starm_solve"]["description"]


@respx.mock
def test_starm_input_formats_are_fetched_once():
    """This runs on the generation path — one probe per server, not per call."""
    route = respx.get("http://localhost:8080/info").mock(
        return_value=httpx.Response(200, json={"task": "sudoku", "input_format": "81 cells"})
    )
    cfg = CareConfig()
    cfg.tools.starm_ports = {"sudoku": 8080}
    for _ in range(3):
        builtin_tools.builtin_tool_specs(cfg.tools)
    assert route.call_count == 1


def test_starm_spec_lists_the_task_ids():
    """A planner can't guess 'game_of_life' from prose about Life."""
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs()}
    description = by_name["starm_solve"]["description"]
    for task in ("sudoku", "maze", "arc", "arithmetic", "game_of_life"):
        assert f"'{task}'" in description, task


def test_starm_spec_forbids_prose_in_problem():
    """The instruction that stops `problem` becoming the user's question."""
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs()}
    description = by_name["starm_solve"]["description"]
    assert "NOT a question" in description
    assert "bbb$ooo$bbb|1" in description  # encoded form shown against the prose one


def test_starm_solve_spec_documents_its_signature():
    by_name = {s["name"]: s for s in builtin_tools.builtin_tool_specs()}
    description = by_name["starm_solve"]["description"]
    assert "starm_solve(problem: str, task_id: str" in description  # no default
    assert "BOTH `problem` AND `task_id` are REQUIRED" in description
    assert "port" in description and "puzzle_id" in description
    assert "algorithmic" in by_name["starm_solve"]["tags"]


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
    "starm_solve",
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
