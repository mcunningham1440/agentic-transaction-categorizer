# Agentic Transaction Categorizer

LLM pipeline that ingests monthly transaction CSVs from multiple banks (Chase, Apple, Venmo, Sam's Club, Ally) and assigns each transaction to one of ~25 spending categories. The agent loop ranks past categorizations by Levenshtein similarity and consults iMessage / Google Calendar / Perplexity to disambiguate edge cases.

## Run

- End-to-end: `.venv/bin/python categorize_transactions.py` — prompts for month/year, reads `data/<m>-<y>/*.csv`, writes `categorized_transactions.csv`.
- Held-out eval: `.venv/bin/python agent_eval.py [--k 4] [--year 2026] [--frac 0.25]` — deterministically holds out a fraction of the first k months' archived transactions, re-categorizes them with those rows hidden from the example pool, and writes `agent_eval_report.csv` (token counts, USD cost, runtime, accuracy, per-tool invocation rate, per-category sensitivity/precision, per-transaction breakdown). Makes live API + tool calls. Costs come from `agent.MODEL_PRICING` — add an entry there when changing `agent.MODEL`, or `compute_cost` raises.
- Smoke-test the archive fetch alone: `.venv/bin/python sheets_archive.py`.
- No automated tests. Exercise changes through one of the entry points above.

## Required setup (per machine, all gitignored)

- `.env` — `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, `VENMO_ACCOUNT_HOLDER_NAME`, `CURRENT_YEAR_ARCHIVE_SHEET_ID`, `PAST_YEAR_ARCHIVE_SHEET_ID`. Template: `.env.sample`.
- `personal_profile.txt` — free-form categorization context fed into the agent's system prompt. Template: `personal_profile.sample.txt`.
- `credentials.json` + `token.json` — Google OAuth (read-only Calendar + Sheets).
- Monthly bank CSVs in `data/<month>-<year>/` following the filename-prefix convention in `transaction_csv_sources.md`.
- **Full Disk Access** for the launching app (Terminal/iTerm/VS Code). The `search_messages` tool reads `~/Library/Messages/chat.db` (`IMESSAGE_DB_PATH` in `tools.py`); without FDA on the app that starts the process, that read fails. Grant it to the app bundle, not to `python`, then relaunch.

## Non-obvious

- **OpenAI Responses API**, not Chat Completions. `agent.py` uses `client.responses.create(...)` with a custom output-extraction protocol (see `_extract_function_calls`, `_serialize_output`). Verify Responses API docs before translating to a different surface.
- **`CATEGORIES` tuple in `tools.py` is the authoritative category list.** Archive rows whose category isn't in the tuple are silently dropped from the similarity pool — intentional, see comment at `categorize_transactions.load_archive`. This also retires prior-year categories that no longer exist.
- **Two archive sheets, combined.** `load_archive()` concatenates `CURRENT_YEAR_ARCHIVE_SHEET_ID` and `PAST_YEAR_ARCHIVE_SHEET_ID`; both are required (the prior-year sheet supplies the trailing-year history early-year transactions need).
- **Trailing 12-month example window.** `agent.categorize_one` restricts each transaction's example pool to the 12 calendar months strictly before its own month (`ARCHIVE_WINDOW_MONTHS`), so the agent never sees same-month/future history. Applies to both production and eval.
- **Archive `Date` is a Sheets serial number.** `sheets_archive.py` reads `A33:E` and converts column A from a days-since-1899-12-30 serial (because of `UNFORMATTED_VALUE`) to a datetime; unparseable dates become `NaT`.
- **No CSV fallback for the archive.** If a sheet ID is wrong or Sheets is unreachable, `load_archive()` raises rather than degrading to a stale local file.
- **OAuth scope broadening triggers auto re-auth.** `google_auth.load_credentials()` detects when `token.json` was issued for a narrower scope set than `SCOPES` and forces a fresh flow instead of 403-ing on the new API later.
- **Sheets API truncates trailing empty cells.** `sheets_archive.py` pads short rows to 5 columns; rows missing `Name` or `Category` are silently skipped as incomplete entries.
