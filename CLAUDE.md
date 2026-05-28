# Agentic Transaction Categorizer

LLM pipeline that ingests monthly transaction CSVs from multiple banks (Chase, Apple, Venmo, Sam's Club, Ally) and assigns each transaction to one of ~25 spending categories. The agent loop ranks past categorizations by Levenshtein similarity and consults iMessage / Google Calendar / Perplexity to disambiguate edge cases.

## Run

- End-to-end: `.venv/bin/python categorize_transactions.py` — prompts for month/year, reads `data/<m>-<y>/*.csv`, writes `categorized_transactions.csv`.
- Smoke-test the archive fetch alone: `.venv/bin/python sheets_archive.py`.
- No automated tests. Exercise changes through one of the entry points above.

## Required setup (per machine, all gitignored)

- `.env` — `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, `VENMO_ACCOUNT_HOLDER_NAME`, `ARCHIVE_SHEET_ID`. Template: `.env.sample`.
- `personal_profile.txt` — free-form categorization context fed into the agent's system prompt. Template: `personal_profile.sample.txt`.
- `credentials.json` + `token.json` — Google OAuth (read-only Calendar + Sheets).
- Monthly bank CSVs in `data/<month>-<year>/` following the filename-prefix convention in `transaction_csv_sources.md`.

## Non-obvious

- **OpenAI Responses API**, not Chat Completions. `agent.py` uses `client.responses.create(...)` with a custom output-extraction protocol (see `_extract_function_calls`, `_serialize_output`). Verify Responses API docs before translating to a different surface.
- **`CATEGORIES` tuple in `tools.py` is the authoritative category list.** Archive rows whose category isn't in the tuple are silently dropped from the similarity pool — intentional, see comment at `categorize_transactions.load_archive`.
- **No CSV fallback for the archive.** If `ARCHIVE_SHEET_ID` is wrong or Sheets is unreachable, `load_archive()` raises rather than degrading to a stale local file.
- **OAuth scope broadening triggers auto re-auth.** `google_auth.load_credentials()` detects when `token.json` was issued for a narrower scope set than `SCOPES` and forces a fresh flow instead of 403-ing on the new API later.
- **Sheets API truncates trailing empty cells.** `sheets_archive.py` pads short rows to 4 columns; rows missing `Name` or `Category` are silently skipped as incomplete entries.
