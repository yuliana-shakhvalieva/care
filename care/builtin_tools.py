"""Bundled standard tools registered into every execution context.

MAGE happily plans chains that call ``web_search``, ``fetch_url``,
``calculator`` and friends — but CARL only knows a tool exists if it
was registered on the :class:`ReasoningContext` before the chain runs.
Out of the box CARE registered *nothing* (the user tools directory
defaults empty), so a generated ``tool`` step died with
``Tool 'web_search' not registered in context``.

This module closes that gap. It ships a small, safe set of standard
tools and registers them by default (gated by
``CareConfig.tools.enable_builtins``). Two seams:

* :func:`register_builtin_tools` — activation. Called by
  :mod:`care.runtime.executor` right after the context is built, so
  every run (chat, CLI, library re-run) sees the same baseline.
* :func:`builtin_tool_specs` — discovery. Returns MAGE-shaped tool
  descriptors so :mod:`care.capability_priming` can tell the planner
  these tools exist (with their call signature in the description),
  keeping the *generated* tool names aligned with the *registered*
  ones.

Design rules for anything added here:

* **Stateless.** CARL deep-copies memory but shares tools across
  parallel steps — a tool with instance state can race.
* **Never raise for an expected "can't do it" case.** Tools return a
  human-readable string the next LLM step can reason about (e.g. a
  missing API key). A raised exception aborts the whole step under the
  default ``RAISE`` recovery policy.
* **Single positional/keyword arg matching the spec.** MAGE maps step
  inputs by name, so the signature here must match the ``description``
  advertised in :func:`builtin_tool_specs`.
"""

from __future__ import annotations

import inspect
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

_log = logging.getLogger("care.builtin_tools")

# Registration-time metadata shared between activation and discovery so
# the two never drift. ``timeout`` is forwarded to ``register_tool``
# (synchronous tools get wrapped in CARL's AsyncToolWrapper; async tools
# manage their own deadline via httpx and ignore it).
_TAGS_WEB = ["information", "external", "web"]
_TAGS_MATH = ["math", "compute"]
_TAGS_TIME = ["time", "utility"]
_TAGS_CODE = ["code", "compute", "external"]
_TAGS_ALGORITHMIC = ["algorithmic", "reasoning", "external"]

# Recency intent (EN + RU) — "latest/newest/current/last…" / "последний/новый…".
_RECENCY_RE = re.compile(
    r"latest|newest|recent|current|\blast\b|\bnow\b|today|this year|nowadays"
    r"|последн|нов(?:ый|ая|ое|ые|инк)|свеж|сейчас|сегодня|текущ|недавн",
    re.I,
)
_YEAR_RE = re.compile(r"\b(20[0-3]\d)\b")


def _bias_recency(query: str) -> str:
    """Bias a web query toward the present for "latest/newest" intent.

    The LLM (training cutoff in the past) tends to bake a stale year into
    "latest X" queries, so web_search returns old results and the synthesis
    step answers with a famous old item. This (a) rewrites any stale
    4-digit year to the current one, and (b) appends the current year when
    the query signals recency but names no year. Non-temporal queries are
    left untouched.
    """
    q = str(query or "").strip()
    if not q:
        return q
    year = datetime.now(timezone.utc).year
    q = _YEAR_RE.sub(lambda m: str(year) if int(m.group(1)) < year else m.group(1), q)
    if str(year) not in q and _RECENCY_RE.search(q):
        q = f"{q} {year}"
    return q


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def _make_web_search(
    provider: str,
    api_key: str | None,
    max_results: int,
    *,
    provider_keys: dict[str, str] | None = None,
) -> Callable[..., Any]:
    """Build the ``web_search`` callable bound to a provider + key(s).

    A factory (rather than a module-level function) so the API key /
    provider from :class:`~care.config.ToolsConfig` is captured at
    registration time instead of read from a global. ``provider_keys``
    holds optional per-provider keys for cross-provider fallback. The
    search is resilient — :func:`_search_resilient` retries transient
    errors and falls back to keyless DuckDuckGo — so a single provider
    blip never kills the step.
    """
    keys = dict(provider_keys or {})

    async def web_search(query: str) -> str:
        """Search the web; returns a numbered list of result snippets."""
        if not str(query or "").strip():
            return "web_search: empty query — pass the search terms as `query`."
        # DuckDuckGo is keyless: a missing primary key is not fatal — fall
        # back to it so web_search works out of the box with zero config.
        primary = provider
        if provider != "duckduckgo" and not api_key and not any(keys.values()):
            primary = "duckduckgo"
        search_q = _bias_recency(query)
        try:
            answer, results = await _search_resilient(
                primary, api_key, search_q, max_results, provider_keys=keys,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't abort the step
            _log.warning("web_search failed (provider=%s): %s", primary, exc)
            return f"web_search error ({primary}): {exc}"
        if not answer and not results:
            return f"No web results found for: {query}"
        blocks: list[str] = []
        if answer:
            # Provider-synthesized answer (Tavily / Serper read page contents
            # server-side) — lead with it so synthesis has the answer up front.
            blocks.append(f"Answer: {str(answer).strip()}")
        seen_urls: set[str] = set()
        rank = 0
        for r in results:
            url = str(r.get("url", "")).strip()
            if url and url in seen_urls:
                continue  # de-dupe repeated URLs (e.g. across fallback engines)
            seen_urls.add(url)
            rank += 1
            blocks.append(
                f"[{rank}] {r.get('title', '').strip()}\n"
                f"{url}\n"
                f"{str(r.get('content', '')).strip()}"
            )
        return "\n\n".join(blocks)

    return web_search


async def _search(
    provider: str,
    api_key: str,
    query: str,
    max_results: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Dispatch ONE search to ``provider``.

    Returns ``(answer, results)``: ``answer`` is a provider-synthesized
    direct answer (Tavily / Serper; ``None`` for engines that don't offer
    one) and ``results`` are normalised ``{"title", "url", "content"}``
    dicts. Raises ``httpx.HTTPStatusError`` on HTTP failure — the resilient
    wrapper :func:`_search_resilient` owns retry + provider fallback.
    """
    import httpx

    timeout = httpx.Timeout(45.0)
    if provider == "tavily":
        async with httpx.AsyncClient(timeout=timeout) as client:
            # "advanced" reads page contents + synthesizes a direct answer,
            # but occasionally 400s on Tavily's side; downgrade to "basic"
            # (cheaper + more reliable) before surfacing the error.
            for depth in ("advanced", "basic"):
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": depth,
                        "search_depth": depth,
                    },
                )
                if resp.status_code == 400 and depth == "advanced":
                    _log.info("tavily 400 on advanced search — retrying basic")
                    continue
                resp.raise_for_status()
                data = resp.json()
                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                    for r in data.get("results", [])
                ]
                return data.get("answer"), results
            return None, []
    if provider == "serper":
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
            # Serper surfaces a direct answer in answerBox / knowledgeGraph.
            box = data.get("answerBox") or {}
            answer = (
                box.get("answer")
                or box.get("snippet")
                or (data.get("knowledgeGraph") or {}).get("description")
            )
            return answer, [
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "content": r.get("snippet", ""),
                }
                for r in data.get("organic", [])
            ]
    if provider == "exa":
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "numResults": max_results,
                    "type": "auto",  # neural when it helps, else keyword
                    "contents": {"text": {"maxCharacters": 1000}},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return None, [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": (r.get("text") or r.get("summary") or ""),
                }
                for r in data.get("results", [])
            ]
    if provider == "duckduckgo":
        return None, await _search_duckduckgo(query, max_results)
    if provider == "serpapi":
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "api_key": api_key,
                    "q": query,
                    "num": max_results,
                    "engine": "google",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return None, [
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "content": r.get("snippet", ""),
                }
                for r in data.get("organic_results", [])
            ]
    if provider == "brave":
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return None, [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("description", ""),
                }
                for r in data.get("web", {}).get("results", [])
            ]
    return None, []


async def _search_duckduckgo(
    query: str, max_results: int
) -> list[dict[str, Any]]:
    """Keyless DuckDuckGo search via the ``ddgs`` library — the universal
    fallback (no API key needed). ``ddgs`` is synchronous, so it runs in a
    worker thread to avoid blocking the event loop."""
    import asyncio

    try:
        from ddgs import DDGS
    except ImportError as exc:  # pragma: no cover - only when dep is absent
        raise RuntimeError(
            "DuckDuckGo search needs the `ddgs` package — `pip install ddgs`"
        ) from exc

    def _run() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return [
                {
                    "title": h.get("title", ""),
                    "url": h.get("href", ""),
                    "content": h.get("body", ""),
                }
                for h in ddgs.text(query, max_results=max_results)
            ]

    return await asyncio.to_thread(_run)


# --- resilience ------------------------------------------------------------
# Fallback chain tried when the primary provider fails. DuckDuckGo is
# keyless, so it's the universal last resort and always available.
_FALLBACK_ORDER: tuple[str, ...] = ("tavily", "serper", "exa", "duckduckgo")
_TRANSIENT_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_WEB_SEARCH_RETRIES = 2  # attempts per provider before falling back
_WEB_SEARCH_BACKOFF_BASE = 0.5  # seconds; doubles each retry


def _is_transient(exc: Exception) -> bool:
    """True for errors worth retrying (timeout / network / 429 / 5xx)."""
    import httpx

    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return False


async def _search_resilient(
    provider: str,
    api_key: str | None,
    query: str,
    max_results: int,
    *,
    provider_keys: dict[str, str] | None = None,
    sleep: Callable[[float], Any] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Run :func:`_search` with retry + provider fallback.

    Tries the primary provider (retrying transient errors with exponential
    backoff), then any other provider that has a configured key, then
    keyless DuckDuckGo — so a transient blip degrades to another engine
    instead of failing the step. Raises the last error only if *every*
    provider failed. ``sleep`` is injectable so tests run without delay.
    """
    import asyncio

    sleep = sleep or asyncio.sleep
    keys = provider_keys or {}

    chain: list[str] = [provider]
    for cand in _FALLBACK_ORDER:
        if cand != provider and cand not in chain:
            chain.append(cand)

    last_exc: Exception | None = None
    for idx, prov in enumerate(chain):
        key = api_key if prov == provider else keys.get(prov, "")
        if prov != "duckduckgo" and not key:
            continue  # no key for this engine — skip it in the fallback chain
        for attempt in range(_WEB_SEARCH_RETRIES):
            try:
                answer, results = await _search(prov, key or "", query, max_results)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_transient(exc) and attempt < _WEB_SEARCH_RETRIES - 1:
                    await sleep(_WEB_SEARCH_BACKOFF_BASE * (2**attempt))
                    continue
                break  # non-transient or out of retries — try next provider
            if answer or results:
                if idx > 0:
                    _log.info("web_search: primary failed; used fallback %s", prov)
                return answer, results
            break  # empty result set — try the next provider
    if last_exc is not None:
        raise last_exc
    return None, []


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\f\v]+")
_BLANKLINES_RE = re.compile(r"\n\s*\n\s*\n+")


def _html_to_text(html: str) -> str:
    """Cheap HTML → text: drop script/style, strip tags, unescape a few
    entities, collapse whitespace. Good enough to feed an LLM step
    without pulling in a parser dependency."""
    import html as _html

    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    text = _html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def _make_fetch_url(max_chars: int) -> Callable[..., Any]:
    """Build ``fetch_url`` bound to a character cap from config."""

    async def fetch_url(url: str) -> str:
        """Fetch a URL and return its readable text (truncated)."""
        u = str(url or "").strip()
        if not u:
            return "fetch_url: empty url — pass the page address as `url`."
        if not u.lower().startswith(("http://", "https://")):
            u = "https://" + u
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0), follow_redirects=True
            ) as client:
                resp = await client.get(
                    u, headers={"User-Agent": "care-agent/1.0 (+fetch_url)"}
                )
                resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            _log.warning("fetch_url failed for %s: %s", u, exc)
            return f"fetch_url error for {u}: {exc}"
        ctype = resp.headers.get("content-type", "")
        body = resp.text
        text = _html_to_text(body) if "html" in ctype.lower() else body.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n…[truncated]"
        return text or f"fetch_url: {u} returned no readable text."

    return fetch_url


# ---------------------------------------------------------------------------
# http_request
# ---------------------------------------------------------------------------


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    """Accept a dict or a JSON-object string (LLMs often emit the latter
    for ``headers`` / ``params`` / ``json_body``). Anything else → None."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _make_http_request(max_chars: int) -> Callable[..., Any]:
    """Build ``http_request`` bound to an output-size cap."""

    async def http_request(
        url: str,
        method: str = "GET",
        headers: Any = None,
        params: Any = None,
        json_body: Any = None,
        data: Any = None,
    ) -> str:
        """Make an HTTP request to any API; returns status + response body."""
        u = str(url or "").strip()
        if not u:
            return "http_request: empty url — pass the endpoint as `url`."
        if not u.lower().startswith(("http://", "https://")):
            u = "https://" + u
        verb = (str(method or "GET").strip().upper()) or "GET"
        import httpx

        kwargs: dict[str, Any] = {}
        if (h := _coerce_mapping(headers)):
            kwargs["headers"] = h
        if (p := _coerce_mapping(params)):
            kwargs["params"] = p
        jb = _coerce_mapping(json_body)
        if jb is not None:
            kwargs["json"] = jb
        elif data not in (None, ""):
            kwargs["content"] = data if isinstance(data, (bytes, str)) else str(data)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0), follow_redirects=True
            ) as client:
                resp = await client.request(verb, u, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _log.warning("http_request %s %s failed: %s", verb, u, exc)
            return f"http_request error ({verb} {u}): {exc}"
        body = resp.text
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n…[truncated]"
        ctype = resp.headers.get("content-type", "")
        return (
            f"HTTP {resp.status_code} {verb} {u}\n"
            f"content-type: {ctype}\n\n{body}"
        ).strip()

    return http_request


# ---------------------------------------------------------------------------
# run_python — sandboxed code execution (the "generate a tool" surface)
# ---------------------------------------------------------------------------


def _make_run_python(
    sandbox_cfg: Any,
    timeout: int,
    max_chars: int,
) -> Callable[..., Any]:
    """Build ``run_python`` bound to a Docker sandbox config.

    Runs the source in a one-shot ``docker run --rm`` container with **no
    host mounts**, capabilities dropped, ``no-new-privileges``, and
    cpu/memory/pids caps from :class:`~care.config.SandboxConfig` — so
    agent-generated code can't touch the host. stdout+stderr are merged
    and returned. Network egress is allowed unless the sandbox policy is
    ``none`` (generated tools usually need to call an API).
    """
    async def run_python(code: str) -> str:
        """Execute Python source in the sandbox; return stdout+stderr."""
        src = str(code or "")
        if not src.strip():
            return "run_python: empty code — pass Python source as `code`."
        return await run_python_source(
            src, sandbox_cfg, timeout=timeout, max_chars=max_chars
        )

    return run_python


async def run_python_source(
    source: str,
    sandbox_cfg: Any,
    *,
    timeout: int = 60,
    max_chars: int = 4000,
) -> str:
    """Run ``source`` in a one-shot Docker sandbox; return stdout+stderr.

    Shared by the ``run_python`` builtin and the on-demand tool
    synthesiser (:mod:`care.tool_synthesis`). Returns a human-readable
    error string (never raises) so a caller/agent can reason about
    failures. The container has no host mounts, ``--cap-drop ALL``,
    ``no-new-privileges``, and the cpu/mem/pids caps from ``sandbox_cfg``;
    network egress is allowed unless the policy is ``none``.
    """
    kind = getattr(sandbox_cfg, "kind", "docker")
    if kind != "docker":
        return (
            "sandbox requires the Docker backend "
            f"(CARE_SANDBOX__KIND=docker); current kind={kind!r}."
        )
    import asyncio
    import shutil
    import uuid

    if shutil.which("docker") is None:
        return (
            "sandbox error: the `docker` CLI isn't on PATH "
            "(is Docker installed and running?)."
        )

    image = getattr(sandbox_cfg, "image", "python:3.12-slim")
    cpus = getattr(sandbox_cfg, "cpu_limit", 2.0)
    mem = getattr(sandbox_cfg, "mem_limit", "1g")
    pids = getattr(sandbox_cfg, "pids_limit", 256)
    net = "none" if getattr(sandbox_cfg, "network_policy", "skill_declared") == "none" else "bridge"
    name = f"care-runpy-{uuid.uuid4().hex[:12]}"
    argv = [
        "docker", "run", "--rm", "-i",
        "--name", name,
        "--cpus", str(cpus),
        "--memory", str(mem),
        "--pids-limit", str(pids),
        "--network", net,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        str(image),
        "python", "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # noqa: BLE001
        return f"sandbox error: couldn't start docker: {exc}"
    try:
        out, _ = await asyncio.wait_for(
            proc.communicate(source.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        await _force_remove_container(name)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return f"sandbox: timed out after {timeout}s (container killed)."
    except Exception as exc:  # noqa: BLE001
        return f"sandbox error: {exc}"
    text = out.decode("utf-8", "replace")
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n…[truncated]"
    return f"[exit {proc.returncode}]\n{text}".strip()


async def _force_remove_container(name: str) -> None:
    """Best-effort ``docker rm -f`` so a timed-out sandbox doesn't leak."""
    import asyncio

    try:
        killer = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(killer.wait(), timeout=10.0)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# starm_solve — one HTTP call to a STARM inference server
# ---------------------------------------------------------------------------


def _coerce_port(value: Any) -> int | None:
    """Parse a port out of whatever the planner passed. ``None`` when it
    isn't a usable TCP port, so the caller can say so instead of quietly
    talking to the wrong server."""
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return port if 0 < port < 65536 else None


def _starm_base_url(host: str, port: int | None, default_port: int) -> str:
    """Compose a STARM server's base URL from the configured host + a port.

    ``CareConfig.tools.starm_host`` carries no port by convention: one STARM
    process serves one checkpoint (= one task), so a box runs several of them
    on neighbouring ports and only the port varies per call. A host that DOES
    carry a port is still honoured — an explicit ``port`` argument overrides
    it, otherwise it stands.
    """
    raw = str(host or "").strip() or "http://localhost"
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    resolved = port or parts.port or default_port
    hostname = parts.hostname or "localhost"
    if ":" in hostname:  # IPv6 literal — urlsplit strips the brackets
        hostname = f"[{hostname}]"
    return urlunsplit(
        (
            parts.scheme or "http",
            f"{hostname}:{resolved}",
            parts.path.rstrip("/"),
            "",
            "",
        )
    )


def _starm_detail(response: Any) -> str:
    """Best-effort human text out of a STARM error response.

    ``/generate`` answers 4xx with ``{"detail": ...}`` — a string for a
    tokenizer/routing error, a list of Pydantic error dicts for a malformed
    body. Both are worth showing verbatim: they are the server's own
    diagnosis of the prompt.
    """
    try:
        detail = response.json().get("detail")
    except Exception:  # noqa: BLE001 — a non-JSON body is still worth showing
        return (response.text or "").strip()
    if isinstance(detail, str):
        return detail
    if detail is None:
        return (response.text or "").strip()
    return str(detail)


def _starm_task_key(task: str) -> str:
    """Fold a task name for port lookup, so ``Game-of-Life`` matches the
    ``game_of_life`` a config was written with. Only the *lookup* is
    lenient — whatever the caller spelled is what goes to the server, whose
    own 422 is the authority on a name it doesn't serve."""
    return str(task or "").strip().lower().replace("-", "_")


#: STARM's task ids, as its tokenizer registry spells them. Named here so a
#: call with no `task` can say what the valid answers are; the server stays
#: the authority on what IT serves (an unknown name comes back as its 422).
_STARM_TASK_IDS: tuple[str, ...] = (
    "sudoku",
    "maze",
    "arc",
    "arithmetic",
    "game_of_life",
)

#: How long the description builder waits on one server's ``/info``. Short:
#: this runs before generation, and a missing format only costs precision.
_STARM_INFO_TIMEOUT = 2.0


@lru_cache(maxsize=8)
def _starm_input_formats(
    host: str, ports: tuple[tuple[str, int], ...]
) -> tuple[tuple[str, str], ...]:
    """Ask each configured STARM server how its task spells a prompt.

    A planner cannot invent a task's encoding — that ``'<pattern>|<step>'``
    or ``'81 cells, . for blanks'`` lives in the checkpoint's tokenizer, and
    guessing it produces a prompt the server rejects. So the format is
    fetched from the servers themselves (``GET /info`` → ``input_format``)
    and quoted into the tool description, which keeps the single copy of it
    on the STARM side.

    Best-effort and cached for the process: a server that's down simply
    contributes no format, and the description falls back to naming the
    task. Never raises — this runs on the generation path.
    """
    import httpx

    formats: list[tuple[str, str]] = []
    for task, port in ports:
        text = ""
        try:
            with httpx.Client(timeout=httpx.Timeout(_STARM_INFO_TIMEOUT)) as client:
                resp = client.get(f"{_starm_base_url(host, port, port)}/info")
                resp.raise_for_status()
                payload = resp.json()
            if isinstance(payload, dict):
                text = str(payload.get("input_format", "")).strip()
        except Exception as exc:  # noqa: BLE001 — advertising is best-effort
            _log.debug("starm /info unavailable for %s on %s: %s", task, port, exc)
        formats.append((task, text))
    return tuple(formats)


def _starm_configured_tasks(tools_cfg: Any | None) -> str:
    """Name this deployment's STARM servers — and how each spells a prompt.

    The planner can only pick a task it knows exists, and the set is
    deployment-specific (``CareConfig.tools.starm_ports``), so it belongs in
    the advertised description rather than in a prompt the user has to write.
    Each task's own input format rides along (see
    :func:`_starm_input_formats`) because naming a task isn't enough: a
    planner that doesn't know the encoding passes the user's question
    through as prose, and the server rejects it.

    Empty string when nothing is configured, leaving the generic blurb.
    """
    ports = dict(getattr(tools_cfg, "starm_ports", None) or {}) if tools_cfg else {}
    if not ports:
        return ""
    known = tuple(sorted(ports.items()))
    lines = [
        f"{task} — {fmt}" if fmt else task
        for task, fmt in _starm_input_formats(
            str(getattr(tools_cfg, "starm_host", "") or "http://localhost"), known
        )
    ]
    return (
        " THIS deployment serves ONLY these tasks — pass one of these exact "
        "strings as `task_id`, with the EXACT input format its `problem` must "
        "use: "
        + "; ".join(lines)
        + ". Encode the user's puzzle into that format yourself — pass the "
        "encoded puzzle ONLY."
    )


def _make_starm_solve(
    host: str,
    ports: dict[str, int] | None,
    default_port: int,
    timeout: float,
) -> Callable[..., Any]:
    """Build ``starm_solve`` bound to the configured STARM deployment.

    STARM (Single Task Algorithmic Reasoning Models) serves one trained
    checkpoint per process over a small HTTP API: ``GET /info`` describes the
    task it serves and the text format that task's prompts take, and
    ``POST /generate`` answers one. This tool is only that call — the task
    list, the prompt formats, the ARC puzzle-id resolution and the answer
    decoding all stay on the server, which is the only place they are
    correct for a given checkpoint.

    ``ports`` maps task -> port (``CareConfig.tools.starm_ports``). Which
    port a task lives on is a property of the deployment, not of the
    question, so a chain names the task and the port is looked up here — no
    user should have to type ``8080`` into a prompt.
    """
    port_map = {_starm_task_key(k): v for k, v in (ports or {}).items()}

    async def starm_solve(
        problem: str,
        # Advertised as required (see `builtin_tool_specs`) but defaulted
        # here on purpose: a planner that still omits it must get a readable
        # instruction back, not a TypeError that aborts the whole step.
        #
        # Named `task_id`, not `task`: a parameter called "task" reads as
        # "what to do", and planners duly filled it with instructions
        # ("Compute next generation") instead of the identifier the server
        # matches on. `task` survives as an alias so already-generated
        # chains keep working.
        task_id: str = "",
        port: Any = None,
        puzzle_id: Any = None,
        task: str = "",
    ) -> str:
        """Solve an algorithmic puzzle with a STARM model; returns its answer."""
        import httpx

        text = str(problem or "").strip()
        served = str(task_id or task or "").strip()
        pid = str(puzzle_id or "").strip()

        known = ", ".join(sorted(port_map)) if port_map else ", ".join(_STARM_TASK_IDS)

        # `task_id` is required, not inferred. It used to be filled in from
        # the server's /info when omitted, which quietly rewarded a planner
        # for leaving it out — and a planner that leaves it out has usually
        # not decided WHICH solver it wants, so the next thing it gets wrong
        # is the encoding. Demanding it keeps the choice explicit.
        if not served:
            return (
                "starm_solve: `task_id` is required — name the STARM task this "
                f"puzzle belongs to. Available: {known}."
            )

        # Reject a description before spending a request on it. A planner
        # that wrote "Compute next generation" here has misread the field,
        # and the fix it needs is the list of ids, not a 422 about a task
        # the server never heard of.
        valid = set(port_map) if port_map else set(_STARM_TASK_IDS)
        if _starm_task_key(served) not in valid:
            return (
                f"starm_solve: {served!r} is not a STARM task id. `task_id` "
                "names WHICH solver to use, not what to do with the puzzle — "
                f"pass exactly one of: {known}."
            )

        resolved_port: int | None = None
        if port not in (None, ""):
            # An explicit port always wins — it's the escape hatch for a
            # server that isn't in the config yet.
            resolved_port = _coerce_port(port)
            if resolved_port is None:
                return (
                    f"starm_solve: {port!r} is not a valid TCP port — pass the "
                    "port the STARM server for this task listens on."
                )
        elif port_map:
            # Safe: an id outside the map was already rejected above, so the
            # lookup cannot miss.
            resolved_port = port_map[_starm_task_key(served)]

        base = _starm_base_url(host, resolved_port, default_port)

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:

            async def describe() -> dict[str, Any]:
                """What this server serves. Best-effort: used to quote the
                prompt format back on a rejection, which isn't worth failing
                the step over."""
                try:
                    resp = await client.get(f"{base}/info")
                    resp.raise_for_status()
                    info = resp.json()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("starm_solve GET %s/info failed: %s", base, exc)
                    return {}
                return info if isinstance(info, dict) else {}

            if not text:
                info = await describe()
                fmt = str(info.get("input_format", "")).strip()
                return (
                    f"starm_solve: empty problem — pass the puzzle as `problem`. "
                    f"{base} serves the "
                    f"{str(info.get('task', '')).strip() or served!r} task"
                    + (f"; its input format is: {fmt}" if fmt else "")
                )

            payload: dict[str, Any] = {"task": served, "input": text}
            if pid:
                payload["puzzle_id"] = pid
            try:
                resp = await client.post(f"{base}/generate", json=payload)
            except Exception as exc:  # noqa: BLE001 — surface, don't abort the step
                _log.warning("starm_solve POST %s/generate failed: %s", base, exc)
                return (
                    f"starm_solve error (POST {base}/generate): {exc}. "
                    "Is a STARM server running on that port?"
                )

            if resp.status_code >= 400:
                detail = _starm_detail(resp)
                hint = ""
                if resp.status_code == 422:
                    # The server rejected the prompt itself — its own
                    # input_format is exactly what the next step needs to fix it.
                    fmt = str((await describe()).get("input_format", "")).strip()
                    if fmt:
                        hint = f" The {served!r} task's input format is: {fmt}"
                return (
                    f"starm_solve: {base} rejected the request "
                    f"(HTTP {resp.status_code}): {detail}{hint}"
                )

            try:
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                return f"starm_solve error: {base} returned a non-JSON body: {exc}"
            if not isinstance(data, dict):
                return f"starm_solve error: unexpected response shape from {base}"

            output = str(data.get("output", "")).strip()
            if not output:
                return (
                    f"starm_solve: {base} returned an empty answer "
                    f"for the {served!r} task."
                )
            steps = data.get("steps")
            max_steps = data.get("max_steps")
            trace = (
                f" in {steps}/{max_steps} recursion steps"
                if steps is not None and max_steps is not None
                else ""
            )
            return f"STARM {served!r}{trace}:\n{output}"

    return starm_solve


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Uses ``simpleeval`` (a CARL dependency) so it's sandboxed against
    arbitrary code execution — only math operators, comparisons and a
    handful of safe functions are allowed.
    """
    expr = str(expression or "").strip()
    if not expr:
        return "calculator: empty expression — pass math as `expression`."
    try:
        from simpleeval import simple_eval
    except Exception:  # noqa: BLE001 — optional dep missing
        return (
            "calculator unavailable: the `simpleeval` package isn't "
            "installed in this environment."
        )
    try:
        return str(simple_eval(expr))
    except Exception as exc:  # noqa: BLE001
        return f"calculator error for {expr!r}: {exc}"


# ---------------------------------------------------------------------------
# current_datetime
# ---------------------------------------------------------------------------


def current_datetime() -> str:
    """Return the current UTC date-time, weekday and long date.

    Gives time-sensitive chains ("today's date", "what day is it?", "is
    this expired?") a ground-truth clock instead of the LLM guessing from
    its (stale) training cutoff. Includes the weekday + long-form date so
    a downstream formatting step doesn't have to derive them.
    """
    now = datetime.now(timezone.utc)
    return (
        f"{now.isoformat(timespec='seconds')} "
        f"({now.strftime('%A')}, {now.strftime('%d %B %Y')})"
    )


# ---------------------------------------------------------------------------
# Activation + discovery
# ---------------------------------------------------------------------------


def register_builtin_tools(
    context: Any,
    tools_cfg: Any,
    sandbox_cfg: Any | None = None,
) -> list[str]:
    """Register the bundled tools onto ``context``.

    Args:
        context: A CARL :class:`ReasoningContext` (duck-typed — needs
            ``register_tool(name, fn, ...)``; ``timeout`` / ``tags``
            keywords are used only when the installed version accepts
            them).
        tools_cfg: A :class:`~care.config.ToolsConfig` (or anything
            exposing the ``web_search_*`` / ``fetch_url_max_chars`` /
            ``enable_code_exec`` / ``code_exec_timeout`` attributes).
        sandbox_cfg: A :class:`~care.config.SandboxConfig` used by
            ``run_python``. ``None`` falls back to ``SandboxConfig()``
            defaults (Docker, ``python:3.12-slim``).

    Returns:
        Names actually registered, in registration order. Per-tool
        failures are logged and skipped rather than raised — one bad
        tool can't block a run.
    """
    if not hasattr(context, "register_tool"):
        _log.warning(
            "context %s lacks register_tool; skipping builtin tools",
            type(context).__name__,
        )
        return []

    if sandbox_cfg is None:
        from care.config import SandboxConfig

        sandbox_cfg = SandboxConfig()

    provider = getattr(tools_cfg, "web_search_provider", "tavily")
    api_key = getattr(tools_cfg, "web_search_api_key", None)
    provider_keys = getattr(tools_cfg, "search_provider_keys", None)
    max_results = int(getattr(tools_cfg, "web_search_max_results", 5) or 5)
    fetch_max = int(getattr(tools_cfg, "fetch_url_max_chars", 4000) or 4000)
    enable_code_exec = bool(getattr(tools_cfg, "enable_code_exec", True))
    code_timeout = int(getattr(tools_cfg, "code_exec_timeout", 60) or 60)
    starm_host = str(getattr(tools_cfg, "starm_host", "") or "http://localhost")
    starm_ports = dict(getattr(tools_cfg, "starm_ports", None) or {})
    starm_port = int(getattr(tools_cfg, "starm_port", 8080) or 8080)
    starm_timeout = float(getattr(tools_cfg, "starm_timeout", 120.0) or 120.0)

    # (name, callable, tags, timeout-seconds)
    specs: list[tuple[str, Callable[..., Any], list[str], float | None]] = [
        (
            "web_search",
            _make_web_search(
                provider, api_key, max_results, provider_keys=provider_keys
            ),
            _TAGS_WEB,
            None,
        ),
        ("fetch_url", _make_fetch_url(fetch_max), _TAGS_WEB, None),
        ("http_request", _make_http_request(fetch_max), _TAGS_WEB, None),
        ("calculator", calculator, _TAGS_MATH, 5.0),
        ("current_datetime", current_datetime, _TAGS_TIME, 5.0),
        (
            "starm_solve",
            _make_starm_solve(
                starm_host, starm_ports, starm_port, starm_timeout
            ),
            _TAGS_ALGORITHMIC,
            None,
        ),
    ]
    if enable_code_exec:
        specs.append(
            (
                "run_python",
                _make_run_python(sandbox_cfg, code_timeout, fetch_max),
                _TAGS_CODE,
                None,
            )
        )

    # ``register_tool``'s signature drifts across mmar_carl versions:
    # 0.2.0 is ``(name, callable)``; newer dev builds add ``timeout`` /
    # ``tags`` keywords. Pass only what the installed version accepts so
    # the same code works against both.
    accepted = _accepted_register_kwargs(context.register_tool)

    registered: list[str] = []
    for name, fn, tags, timeout in specs:
        kwargs: dict[str, Any] = {}
        if "timeout" in accepted and timeout is not None:
            kwargs["timeout"] = timeout
        if "tags" in accepted and tags:
            kwargs["tags"] = tags
        try:
            context.register_tool(name, fn, **kwargs)
            registered.append(name)
        except Exception as exc:  # noqa: BLE001
            _log.warning("failed to register builtin tool %r: %s", name, exc)
    _log.info("registered %d builtin tools: %s", len(registered), ", ".join(registered))
    return registered


def _accepted_register_kwargs(register_tool: Callable[..., Any]) -> set[str]:
    """Optional keyword names ``register_tool`` accepts.

    Returns ``{"timeout", "tags"}`` when the callable takes ``**kwargs``
    or names them explicitly; an empty set for the minimal
    ``(name, callable)`` signature shipped in mmar_carl 0.2.0. Falls back
    to "accept both" if the signature can't be introspected (C-level or
    exotic callables) — a wrong guess is caught by the per-tool
    ``try/except`` above.
    """
    optional = {"timeout", "tags"}
    try:
        params = inspect.signature(register_tool).parameters
    except (TypeError, ValueError):
        return optional
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return optional
    return optional & set(params)


def builtin_tool_specs(tools_cfg: Any | None = None) -> list[dict[str, Any]]:
    """MAGE-shaped descriptors for the bundled tools.

    Shape matches ``care.capability_priming._tool_dict`` (``name`` /
    ``source`` / ``description`` / ``tags``). The call signature is baked
    into ``description`` so the planner emits an ``input_mapping`` whose
    key matches the tool's parameter name.

    ``run_python`` is advertised only when ``tools_cfg.enable_code_exec``
    is truthy (default on). Other bundled tools are advertised regardless
    of whether a key is set (``web_search`` degrades to a "not configured"
    line at call time rather than vanishing from the catalogue).
    """
    specs = [
        {
            "name": "web_search",
            "source": "care:builtin",
            "description": (
                "web_search(query: str) -> str. Search the web for current "
                "information and return a numbered list of result snippets "
                "(title, url, content). Pass the search terms or the user's "
                "question as `query`. ALWAYS call this (never answer from "
                "memory) for anything time-sensitive or outside the model's "
                "knowledge: news, weather, prices, sports, recent events, docs. "
                "For 'latest/newest/current' queries it biases results to the "
                "current year — when summarizing, PREFER the most recent dated "
                "result, don't default to a famous older one."
            ),
            "tags": list(_TAGS_WEB),
        },
        {
            "name": "fetch_url",
            "source": "care:builtin",
            "description": (
                "fetch_url(url: str) -> str. Download a web page and return "
                "its readable text (HTML stripped, truncated). Pass the page "
                "address as `url`. Pair with web_search to read a result."
            ),
            "tags": list(_TAGS_WEB),
        },
        {
            "name": "http_request",
            "source": "care:builtin",
            "description": (
                "http_request(url: str, method='GET', headers=None, "
                "params=None, json_body=None) -> str. Make an HTTP request to "
                "any REST API and return the status + response body. Use to "
                "call an API directly (weather, finance, GitHub, …). "
                "`headers`/`params`/`json_body` take a JSON object."
            ),
            "tags": list(_TAGS_WEB),
        },
        {
            "name": "calculator",
            "source": "care:builtin",
            "description": (
                "calculator(expression: str) -> str. Evaluate an arithmetic "
                "expression (e.g. '2 * (3 + 4)') and return the result. Pass "
                "the math as `expression`."
            ),
            "tags": list(_TAGS_MATH),
        },
        {
            "name": "current_datetime",
            "source": "care:builtin",
            "description": (
                "current_datetime() -> str. Return the current UTC date-time, "
                "weekday and long date. Takes no arguments. ALWAYS use this "
                "(never answer from memory) for ANY question about the current "
                "date, time, day of week, month, or year — the model's own "
                "sense of 'today' is stale and will be WRONG."
            ),
            "tags": list(_TAGS_TIME),
        },
        {
            "name": "starm_solve",
            "source": "care:builtin",
            # The task ids are named here so a planner emits a valid one
            # instead of inventing "Game of Life". They're a hint, not a
            # contract: the server stays the authority (a name it doesn't
            # serve comes back as its own 422), and a configured deployment
            # narrows the list to the servers it actually runs, below.
            "description": (
                "starm_solve(problem: str, task_id: str, port=None, "
                "puzzle_id=None) -> str. BOTH `problem` AND `task_id` are "
                "REQUIRED — a call that omits `task_id` fails without "
                "reaching the solver. Solve one ALGORITHMIC puzzle with a "
                "STARM model — a small recurrent solver that follows an "
                "algorithm exactly, where an LLM guesses. "
                "`task_id` selects WHICH solver runs. It is NOT a "
                "description of the work: 'Compute next generation', 'solve "
                "it' or any sentence is wrong and is rejected. It is a STARM "
                "task id, spelled EXACTLY as one of: "
                "'sudoku' (fill a 9x9 grid), 'maze' (shortest path through a "
                "grid), 'arc' (ARC-AGI abstract grid transformation), "
                "'arithmetic' (recover the operators of an expression), "
                "'game_of_life' (advance a Life pattern N generations). "
                "Lower-case with underscores — 'Game of Life' or 'sudoku "
                "puzzle' are not task ids. A deployment serves only the "
                "subset it runs servers for; that subset, and each one's "
                "input format, is listed below when configured. "
                "CRITICAL: `problem` is NOT a question and NOT a sentence — it "
                "is the puzzle ENCODED in its task's own text format, and "
                "nothing else. Strip every word of the user's phrasing, "
                "extract the puzzle data, and re-encode it. Passing prose "
                "(\'what happens to bbb$ooo$bbb after 1 generation?\') is "
                "always wrong; the encoded form (\'bbb$ooo$bbb|1\') is what "
                "the server parses. "
                "EVERY number the user states — how many generations to "
                "advance, which step to reach, the target value — is part "
                "of `problem`, not a separate argument and not something "
                "to drop: there is nowhere else to put it. 'bbb$ooo$bbb' "
                "alone is INCOMPLETE and the server rejects it; "
                "'bbb$ooo$bbb|1' is the same puzzle carrying the count "
                "the user asked for. Copy the shape of the example in the "
                "task's format below exactly — every separator ('|', "
                "'$', '=') and any trailing number it shows. "
                "Each STARM server serves ONE task on its own port, so "
                "`task_id` picks which solver answers and the port is "
                "configured, not "
                "guessed; `port` overrides it only for a server not yet in the "
                "config. `puzzle_id` is ARC-only and required there (e.g. "
                "\'007bbfb7\'). Calling with an empty `problem` returns the "
                "server\'s own format description."
                + _starm_configured_tasks(tools_cfg)
            ),
            "tags": list(_TAGS_ALGORITHMIC),
        },
    ]
    enable_code = True if tools_cfg is None else bool(
        getattr(tools_cfg, "enable_code_exec", True)
    )
    if enable_code:
        specs.append(
            {
                "name": "run_python",
                "source": "care:builtin",
                "description": (
                    "run_python(code: str) -> str. Execute Python source in "
                    "an isolated Docker sandbox and return its stdout/stderr. "
                    "Use to synthesise a tool on the fly: write code that "
                    "calls an API, parses data, or computes something the "
                    "other tools don't cover. Network egress and the standard "
                    "library are available; print the result to stdout."
                ),
                "tags": list(_TAGS_CODE),
            }
        )
    return specs


__all__ = [
    "builtin_tool_specs",
    "calculator",
    "current_datetime",
    "register_builtin_tools",
    "run_python_source",
]
