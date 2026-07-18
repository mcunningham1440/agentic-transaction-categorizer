# Transaction categorizer

Agentic workflow for intelligently categorizing monthly transactions.

## Download transaction CSVs

The supervised browser downloader currently supports Chase and is structured as
a source registry so additional banks can be added independently. It opens a
visible persistent Chromium profile, pauses for human MFA/CAPTCHA completion,
and saves a validated CSV into the folder expected by the categorization
pipeline.

One-time setup:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

Add `CHASE_URL`, `CHASE_USERNAME`, and `CHASE_PASSWORD` to `config/.env` as
shown in `config/.env.sample`. Then download the previous month's activity:

```bash
.venv/bin/python download_transactions.py chase
```

Or choose an explicit month/year:

```bash
.venv/bin/python download_transactions.py chase --month 6 --year 2026
```

Chase exports year-to-date activity; `categorize_transactions.py` later filters
it to the chosen month. The command refuses to add a second Chase CSV to an
existing month folder unless `--replace` is passed. A replacement is deleted
only after the new CSV passes header validation.

The browser profile is stored in the gitignored `.pw-chase-profile/` directory
so Chase can remember device trust. Password entry is local: the model sees and
types the literal `ACCOUNT_PASSWORD` sentinel, which is substituted only while a
password input has focus. The Chase adapter handles its stable mobile-app MFA
chooser directly, sends the push, then sounds a terminal bell and pauses for you
to approve it on your phone. If Chase shows multiple devices, set the optional
`CHASE_MFA_DEVICE` value to the device name displayed by Chase.

The default browser model is `gpt-5.6-luna` with low reasoning for lower
latency. Set `BROWSER_AUTOMATION_MODEL` in `config/.env` to override it. Stable
Chase login and mobile-MFA screens are handled locally without model calls; the
model is reserved for the dynamic account-activity and download screens.
