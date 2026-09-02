# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What CARE is

CARE — Collaborative Agent Reasoning Ecosystem — is a Textual TUI + headless CLI (`maestro`, with a `care` back-compat alias) built on top of a four-module stack:

- **MAGE** (`mmar-mage`, **required** dep) — turns a query into a CARL chain.
- **CARL** (`mmar-carl`, optional `care[carl]` extra) — runs CARL chains; sandbox runtime; preflight.
- **GigaEvo Memory** (`gigaevo-client`, required dep) — entities (chain / agent / agent_skill / memory_card), library, SSE.
- **GigaEvo Platform** — GA over chains, accept-and-promote.

All upstreams resolve from PyPI (there is no longer a `[tool.uv.sources]` block pinning sibling checkouts — `docs/ARCHITECTURE.md` still links `../carl-mage` etc. as prose references only). Everything except `mmar-mage` / `gigaevo-client` is imported **lazily** (inside the function that needs it) so a minimal install boots the CLI and a missing extra surfaces as a friendly install hint instead of an `ImportError` at startup. `tests/test_uvx_smoke.py` is the guard: it boots `--help` against wheel-only deps to catch an accidental top-level import of an optional extra.

## Commands

```bash
make run                       # launch the TUI (uv sync + uv run care)
make run LOG=1 LOG_LEVEL=DEBUG # also write logs/care-ui-* + logs/care-app-* sidecars
make test                      # uv run pytest (whole suite)
make lint                      # uv run ruff check .
uv sync --extra dev            # install dev deps (pytest, pytest-asyncio, ruff, respx) — needed before `pytest`
uv run maestro --help          # CLI surface
```

Focused test runs:

```bash
uv run --extra dev pytest tests/test_chat_revise.py -q                    # one file
uv run --extra dev pytest tests/test_cli.py::TestGenerateCommand -q       # one class
uv run --extra dev pytest tests/test_cli.py -k "revise" -q                # by keyword
uv run --extra dev pytest tests/test_screen_library.py -x -q              # fail fast
```

The suite is ~209 files / ~71k lines, dominated by `tests/test_cli.py` (~4k lines) and the `tests/test_screen_*.py` family. Chat coverage is **split across many small files** (`tests/test_chat_*.py`, `tests/test_widget_chat_input.py`, `tests/test_chat_slash_command_manifest.py`) rather than one monolith. Run a focused slice while iterating, then the full suite before declaring done.

Sandbox runs of `Bash` sometimes can't reach pypi — `uv sync` will succeed only once. After it has, `uv run --extra dev pytest …` works offline.

**CI runs only a narrow slice** (`tests/test_packaging.py` + `tests/test_screen_save_report.py`, plus ruff) because upstream resolution isn't guaranteed on a clean runner. A green CI is *not* evidence the suite passes — run it locally.

## High-level architecture

```
care/
├── app.py                 CareApp Textual entry; global bindings (Ctrl+P/Ctrl+B/Ctrl+K/Ctrl+S/Ctrl+R/Ctrl+Q, Ctrl+C=quit)
├── cli.py                 `maestro` headless CLI router + every subcommand (~4.5k lines, argparse subparsers)
├── config.py              Pydantic CareConfig; nested mage/memory/platform/hub/upload/sandbox/tools/
│                          telemetry/defaults/chat/context/artifacts
├── memory.py              CareMemory facade over GigaEvoClient (stamps CareChainMetadata)
├── platform.py            CarePlatform facade for evolution
├── builtin_tools.py       Tools registered into every run context: web_search, fetch_url, http_request,
│                          calculator, current_datetime, starm_solve, run_python
│                          (gated by tools.enable_builtins)
├── tool_synthesis.py      Self-healing missing tools: disk cache → Memory → LLM-synthesise (sandboxed)
├── capability_priming.py  Tells MAGE about locally-installed skills/MCP/tools before planning
├── runtime/               ~70 adapters, one rule per file (upstream callback → Textual Message, etc.)
│   ├── i18n.py               t(key) lookup over locales/{en,ru}.json — see "Localization" below
│   ├── mage_poster.py        MAGE progress → StageStarted/StageCompleted/StageProgress messages
│   ├── carl_streamer.py      CARL run callbacks → StepStarted/StepCompleted/Progress/ChainCompleted
│   ├── executor.py           build_run_context / execute_chain_async (fresh / re-run / replay-override)
│   ├── llm_client.py         build_llm_client / build_carl_llm_client (see "MAGE vs CARL clients")
│   ├── keystore.py           keystore:// secret URLs backed by macOS Keychain / secret-service
│   ├── artifacts.py          Copies sandbox output_files out to ~/.care/artifacts
│   ├── theme.py              Theme registry; the `$accent` brand colour (`_LIGHT_VARS`/`_DARK_VARS`)
│   ├── clipboard.py          copy_text — OSC-52 + pbcopy/xclip/wl-copy fallback
│   ├── cancellation.py       CancellationToken / CancellationGroup
│   └── …
├── screens/               50+ Textual screens/modals (chat, library, evolution, artifacts, inspection, …)
│   └── chat.py              PRIMARY user surface — ChatScreen owns ~75% of user-facing behaviour (~18k lines)
├── sandbox/               AgentSkill sandbox backends (local / docker / e2b / firejail) + trust, audit,
│                          network_policy, output_mediation, resources
└── assets/                Packaged static assets (airi_logo_8/10/12/16.png + the un-suffixed default)
```

`docs/ARCHITECTURE.md` is the layer-by-layer walkthrough (generation / execution / persistence / evolution) and is **drift-guarded** by `tests/test_architecture_doc.py`. The four canonical screens are **Chat → Artifacts | Library | Evolution**; the legacy `QueryScreen`/`GenerationScreen`/`InspectionScreen` still exist for non-chat boot + drill-down paths.

### ChatScreen is the centre of gravity

`care/screens/chat.py` is the **only** screen most users see; new features land there first. Key conventions:

- **`_post_line(role, text, *, severity=None, provenance=None, chrome=False, extra_class=None, linkify_commands=False, rich_override=None)`** is the single chat-line mounter. `chrome=True` skips the `[HH:MM] role` caption (boot banner, welcome lines, mode-flip hints). `extra_class` adds a per-call CSS hook (e.g. `chat-line-pre-answer` pads before the assistant answer). `severity` mirrors `WARNING`/`ERROR` into `care.chat` logging. It marshals back to the main thread via `call_from_thread` when called off-loop, so worker code can post safely.
- **Roles** (`ChatRole = Literal["user", "assistant", "system", "tool"]`) gate rendering: `assistant` + `system` mount as `Markdown`, `user` + `tool` as `Static`. The Markdown widget's inherited `padding: 0 2` is stripped via `ChatScreen Markdown.chat-line { padding: 0; }`.
- **Stage trail**: MagePoster events post `▶ Friendly Label…` (tool line) → `_stage_started_indexes[stage]` records the widget index → `StageCompleted` mutes the matching line via `_STAGE_DONE_CSS_CLASS`. Sub-rows use `  ⎿ <text>`.
- **Modes + pipeline**: `ChatMode = Literal["interactive", "production"]` (legacy `ad_hoc` still deserializes — `normalise_mode()` maps it to `interactive`). Both modes run one pipeline `GENERATE → PREVIEW → RUN → SAVE → BASELINE → EVOLVE` (`Stage` StrEnum). GENERATE/PREVIEW are always auto; only RUN/SAVE/BASELINE/EVOLVE carry a `StagePolicy` (`auto`/`ask`/`skip`) from the mode's `ModeSpec` in `MODE_SPECS`, overridable via `CARE_CHAT__MODE__<MODE>__<STAGE>` and merged by `resolve_mode_spec`. **Interactive** = `run=ask`, save/baseline/evolve `skip` (save is a chain-action button, not a modal), `followup="reuse"`. **Production** = save/baseline/evolve `auto`, driven by `_drive_production_pipeline` (`_resolve_stage` per stage), `followup="revise"` (a follow-up prompt is treated as a `/revise` of the saved agent). `_render_pipeline_strip` renders the live `◆/○/◇/✗` cells plus the "thinking…" spinner on **one** `#chat-pipeline-strip` line; `_resolve_stage` emits a `chat.pipeline.stage` telemetry event per stage.
- **Slash commands** are registered by module-level `@_register("name")` decorators at the *bottom* of chat.py into `_COMMAND_HANDLERS` (~43 handlers). Every command must also appear in `_COMMAND_BLURBS` and in `docs/screens/README.md` — `tests/test_chat_slash_command_manifest.py` and `tests/test_screens_index_doc.py` fail the build on drift, and `tests/test_i18n.py` pins `_COMMAND_BLURBS` against `locales/en.json`.
- **Interactive context**: `_interactive_history` keeps user/assistant turns across generations (gated on `spec.followup == "reuse"`). `/new`, `/clear`, and mode flips reset it. Capped via `CARE_CHAT__AD_HOC_HISTORY_TURNS` (default 6) and `CARE_CHAT__AD_HOC_HISTORY_CHARS` (default 1200/turn) — env keys keep the legacy `AD_HOC` spelling.
- **Generation retry**: `_generate_with_retry` runs MAGE generation up to `CARE_CHAT__GENERATION_MAX_ATTEMPTS` (default 3) times with exponential backoff. Cancellation re-raises immediately.
- **`/revise` (NL chain edit)**: `_cmd_revise` → `_run_edit` worker drives MAGE's `MAGEGenerator.edit(...)` via `care.generation.run_edit` (`save=False`), previews with `care.runtime.chain_edit_view.render_edit_plan_lines`, then `ConfirmModal` → saves a NEW VERSION via `app.memory.save_chain(entity_id=…)` (preserving CARE metadata stamping). Forms: `/revise <id> <change>` or bare `/revise <change>` (MAGE resolves the chain). The library's `R` row-action hands off by seeding `ChatScreen.seed_input("/revise <id> ")`. Editing is **explicit** — plain prose is intentionally NOT auto-routed into edit mode.
- **Interactive answer synthesis**: with ≥2 successful steps, `_synthesise_user_answer` makes one extra LLM call merging step outputs into a coherent reply (bracketed by `▶/✓ Synthesising answer` tool lines). Production skips this — chains there must be reproducible.
- **`@<path>` file refs**: `_resolve_file_refs` greedily extends across whitespace while each candidate path keeps resolving on disk, so `@../My Notes.md` works unquoted. Also honors `@"…"`/`@'…'`, `.pdf` via pypdf, office/rich-text extraction (`.docx`/`.pptx`/`.xlsx`/`.odt`/`.rtf` → `care/runtime/document_extract.py`, routed via `_read_document_ref`; legacy `.doc`/`.ppt`/`.xls` get a "re-save as …" hint), and image base64-embed for vision models. Doc refs share the PDF two-cap pattern (`CARE_CHAT__DOC_REF_MAX_BYTES` 25 MB on-disk / `CARE_CHAT__DOC_TEXT_MAX_CHARS` 200 k extracted).

### Localization (easy to break)

The TUI is bilingual and **defaults to Russian** (`config.defaults.ui_language = "ru"`). User-facing strings go through `care.runtime.i18n.t("dotted.key")` with catalogs in `care/runtime/locales/{en,ru}.json`; lookup falls back active → English → the key itself. `tests/test_i18n.py` pins that both catalogs share an identical key set, so **a new string must be added to both files**. Write natural, idiomatic Russian — never a word-for-word calque. This is the *interface* language only; the *agent's answer* language is the separate `config.defaults.language`, forwarded to CARL.

### Configuration precedence (low → high)

1. Defaults in `care.config.CareConfig`
2. `~/.config/care/config.toml` (or `~/.config/care/profiles/<name>.toml` when `CARE_PROFILE` is set)
3. `./care.toml`
4. `CARE_*` env vars (nested via `__`: `CARE_MAGE__MODEL`, `CARE_CHAT__DEFAULT_MODE`, `CARE_CHAT__MODE__INTERACTIVE__RUN`, …)

`.env.example` documents the full surface and is **symmetry-tested** against `CareConfig` by `tests/test_env_example.py` — every nested config field needs at least a commented-out stub there, and every stub must map to a real Pydantic field. `maestro init [--non-interactive]` writes a starter `.env`; `maestro doctor` reports resolved config + live probes; `maestro migrate-secrets` rewrites literal API keys as `keystore://…` URLs. First boot without `~/.config/care/config.toml` lands on `SettingsScreen`; returning users go to ChatScreen.

### Themes / brand color

`$accent` (user-message tint, the `>` chat prompt, mode-toggle active state, every screen's hot-element accent) is defined once in `care/runtime/theme.py`'s `_LIGHT_VARS["accent"]` and `_DARK_VARS["accent"]` (currently the same `#2ebfae` in both). Re-brand by changing those two hex strings — screens reference the design token rather than hard-coding.

### MAGE vs CARL clients

CARE talks to MAGE through the raw `openai.OpenAI` SDK (`runtime/llm_client.build_llm_client`). CARL's step executors call `get_response_with_retries(prompt, retries)`, which the raw SDK doesn't expose — use `build_carl_llm_client(config, token_counter=…)` instead. The token-counter variant intercepts `chat.completions.create` responses and folds `response.usage` into a `SessionTokenCounter` so the StatusBar shows real numbers (without it, chat reads `in 0 / out 0` for CARL runs).

## Testing conventions

- **Async by default**: `pyproject.toml` sets `asyncio_mode = "auto"` and `asyncio_default_fixture_loop_scope = "function"`.
- **`tests/conftest.py` autouse fixtures** apply to every test: Textual animations are forced to `"none"` (so a pilot never observes a widget mid-tween), the persisted `LibraryScreen` view-state sidecar is redirected to `tmp_path` via `CARE_VIEW_STATE_PATH`, and the UI language is pinned to **English** (screen tests assert English copy). Extend these instead of bypassing them when adding new on-disk state.
- **Screen-test scaffold pattern**: a minimal `class _Host(App)` that sets `self.config = CareConfig()` and does `push_screen(ChatScreen())` in `on_mount`, plus a `_chat(app)` helper that walks `app.screen_stack`. ~41 test files already use this shape (`tests/test_chat_status_strip.py` is a clean, short example) — copy it rather than inventing a new host. Variants (`_MemHost`, `_ProdHost`) pre-populate `app.memory`/`app.config`. Existing monkeypatch sites (`care.runtime.llm_client.build_llm_client`, `care.runtime.clipboard.copy_text`, MAGE/CARL stubs) are already wired.
- **`Input` vs `ChatInput`**: tests query the prompt as `screen.query_one("#chat-input", ChatInput)`. The widget subclasses `TextArea` but exposes `.value` / `.cursor_position` / `action_submit` and posts `Input.Submitted` for back-compat — use those, not raw TextArea methods.
- **Verifying widget classes after `_post_line`**: in some scaffolds the mount is async and `query_one("#chat-line-N")` raises `NoMatches` before the next `pilot.pause()`. Await another pause, or monkeypatch `_post_line` to capture call args.
- **Doc/config guard tests** — expect these to fail when you add a surface without its paperwork: `test_env_example.py` (config ↔ `.env.example`), `test_screens_index_doc.py` (screens + slash commands ↔ `docs/screens/README.md`, regenerate with `scripts/generate_screens_index.py`), `test_architecture_doc.py`, `test_chat_slash_command_manifest.py`, `test_i18n.py`, `test_ci_workflows.py`, `test_packaging.py` (rebuilds the wheel — slow).

## Things to avoid

- **Don't re-instate `pillow` as a top-level dep** — `rich-pixels>=3.0` already pins `pillow>=10.0.0`; the duplicate was deliberately removed.
- **Don't bind `Ctrl+C` at the screen level** — the app-level `Binding("ctrl+c", "global_quit", …, priority=True)` in `care/app.py` owns it for the classic quit chord. The chat surface intentionally claims only `super+c` for copy-selection (drag-select already auto-copies via `on_text_selected`); `ctrl+y` copies the last reply, `ctrl+shift+y` the transcript.
- **Don't post the MAGE metadata summary as an `assistant` line** — it rides as `⎿`-prefixed tool sub-rows under `✓ Describing steps` so it groups with the stage trail. Use `_format_result_summary_rows(result)` (returns `list[str]`) and emit each row as `self._post_line("tool", f"  ⎿ {row}")`. The legacy `_format_result_summary(result) → str` is kept only for the Production-save path.
- **Don't add a hard-coded English string to a screen** — route it through `t(...)` and add both catalog entries (see Localization).
- **Don't ship generated tool code outside the sandbox** — `care/tool_synthesis.py` only ever executes LLM-written code via `care.builtin_tools.run_python_source` (Docker sandbox), never in CARE's process.
- **Branch discipline**: `.github/workflows/mirror.yml` force-pushes an **orphan squashed snapshot** of the tree to the public mirror on every push to `release`. Private history must never reach `release` expecting to be preserved, and `main` is the working branch.

## Logging

`make run LOG=1` writes two side-by-side sidecars per launch:

- `logs/care-ui-<timestamp>.log` — Textual UI events (compose, mount, dispatch, render), driven by `TEXTUAL_LOG`.
- `logs/care-app-<timestamp>.log` — Python app/client log (every `care.*` module + httpx Memory/Platform + MAGE/CARL workers), driven by `CARE_LOG_FILE` / `CARE_LOG_LEVEL`.

`_post_line` mirrors every chat entry into `care.chat` at INFO (`tool` rides DEBUG, severity-tagged lines ride WARNING/ERROR), so a session transcript survives in the app log without extra instrumentation.
