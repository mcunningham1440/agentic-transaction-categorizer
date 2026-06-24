# Agentic Transaction Categorizer

LLM pipeline that ingests monthly transaction CSVs from multiple banks (Chase, Apple, Venmo, Sam's Club, Ally) and assigns each transaction to one of ~25 spending categories. The agent loop ranks past categorizations by Levenshtein similarity and consults iMessage / Google Calendar / Perplexity to disambiguate edge cases.

## Layout

Code lives in the `categorizer/` package; the two root scripts
(`categorize_transactions.py`, `agent_eval.py`) are thin wrappers over
`categorizer.pipeline.main` / `categorizer.eval.main`. Paths are resolved
through `categorizer/paths.py` (`PROJECT_ROOT`, `CONFIG_DIR`, `OUTPUT_DIR`,
`DATA_DIR`). Hand-edited/secret inputs live in `config/`, generated artifacts in
`output/`, monthly bank CSVs in `data/`. `categorizer/__init__.py` calls
`load_dotenv(config/.env)` on import, so env vars are populated before any
submodule reads `os.environ`. Subpackages: `categorizer/archive/`
(`sheets.py`, `index.py`, `google_auth.py`) and `categorizer/tools/`
(`messages.py`, `calendar.py`, `web.py`, with `__init__.py` holding
`TOOL_DEFINITIONS` + `dispatch_tool`); `categorizer/categories.py` is the
category loader.

## Run

- End-to-end: `.venv/bin/python categorize_transactions.py` — prompts for month/year, reads `data/<m>-<y>/*.csv`, writes `output/categorized_transactions.csv`.
- Held-out eval: `.venv/bin/python agent_eval.py [--k 4] [--year 2026] [--frac 0.25]` — deterministically holds out a fraction of the first k months' archived transactions, re-categorizes them with those rows hidden from the example pool, and writes `output/agent_eval_report.csv` (token counts, USD cost, runtime, accuracy, per-tool invocation rate, per-category sensitivity/precision, per-transaction breakdown). Makes live API + tool calls. Costs come from `categorizer.agent.MODEL_PRICING` — add an entry there when changing `agent.MODEL`, or `compute_cost` raises.
- Smoke-test the archive fetch alone: `.venv/bin/python -m categorizer.archive.sheets`.
- No automated tests. Exercise changes through one of the entry points above.

## Required setup (per machine, all gitignored; live in `config/`)

- `config/.env` — `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, `VENMO_ACCOUNT_HOLDER_NAME`, `CURRENT_YEAR_ARCHIVE_SHEET_ID`, `PAST_YEAR_ARCHIVE_SHEET_ID`. Template: `config/.env.sample`.
- `config/categories.yaml` — authoritative category list plus optional per-category `instructions`. Template: `config/categories.sample.yaml`.
- `config/credentials.json` + `config/token.json` — Google OAuth (read-only Calendar + Sheets).
- Monthly bank CSVs in `data/<month>-<year>/` following the filename-prefix convention in `config/transaction_csv_sources.md`.
- **Full Disk Access** for the launching app (Terminal/iTerm/VS Code). The `search_messages` tool reads `~/Library/Messages/chat.db` (`IMESSAGE_DB_PATH` in `categorizer/tools/messages.py`); without FDA on the app that starts the process, that read fails. Grant it to the app bundle, not to `python`, then relaunch.

## Non-obvious

- **OpenAI Responses API**, not Chat Completions. `categorizer/agent.py` uses `client.responses.create(...)` with a custom output-extraction protocol (see `_extract_function_calls`, `_serialize_output`). Verify Responses API docs before translating to a different surface.
- **`config/categories.yaml` is the authoritative category list (gitignored; template `config/categories.sample.yaml`).** `categorizer.categories._load_categories()` reads it at import into `CATEGORIES` (tuple of names) and `CATEGORY_INSTRUCTIONS` (`{name: optional per-category guidance}`); the loader raises on a missing/malformed file or duplicate names rather than degrading. Each entry's `instructions` (optional, defaults to `""`) is appended to that category's bullet in the agent prompt. Archive rows whose category isn't in `CATEGORIES` are silently dropped from the similarity pool — intentional, see comment at `categorizer.pipeline.load_archive`. This also retires prior-year categories that no longer exist. Renaming a `name` orphans existing archive rows, so the `acessories` typo is kept deliberately.
- **Two archive sheets, combined.** `load_archive()` concatenates `CURRENT_YEAR_ARCHIVE_SHEET_ID` and `PAST_YEAR_ARCHIVE_SHEET_ID`; both are required (the prior-year sheet supplies the trailing-year history early-year transactions need).
- **Trailing 12-month example window.** `categorizer.agent.categorize_one` restricts each transaction's example pool to the 12 calendar months strictly before its own month (`ARCHIVE_WINDOW_MONTHS`), so the agent never sees same-month/future history. Applies to both production and eval.
- **Archive `Date` is a Sheets serial number.** `categorizer/archive/sheets.py` reads `A33:E` and converts column A from a days-since-1899-12-30 serial (because of `UNFORMATTED_VALUE`) to a datetime; unparseable dates become `NaT`.
- **No CSV fallback for the archive.** If a sheet ID is wrong or Sheets is unreachable, `load_archive()` raises rather than degrading to a stale local file.
- **OAuth re-auth is automatic in two cases.** `categorizer.archive.google_auth.load_credentials()` forces a fresh browser flow (1) when `config/token.json` was issued for a narrower scope set than `SCOPES` (instead of 403-ing on the new API later), and (2) when refreshing a stored token raises `RefreshError`/`invalid_grant` (revoked or expired refresh token — OAuth clients in "Testing" status expire refresh tokens after 7 days); the dead `token.json` is deleted and the flow restarts. Caveat: this hangs in headless/scheduled runs since `run_local_server` waits for a browser.
- **Sheets API truncates trailing empty cells.** `categorizer/archive/sheets.py` pads short rows to 5 columns; rows missing `Name` or `Category` are silently skipped as incomplete entries.
