#!/bin/zsh
# Daily after-close job: refresh market series and snapshot option chains.
# yfinance has no option history, so missing a day means that day's IV is lost.
# Schedule with launchd (see scripts/com.llmmoney.daily.plist) or cron, e.g. weekdays 16:30 ET:
#   30 16 * * 1-5  /path/to/LLMmoney/scripts/daily_record.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
{
  echo "=== $(date) ==="
  .venv/bin/llmmoney ingest-market --start 2025-01-01
  .venv/bin/llmmoney record-options
} >> data/logs/daily_record.log 2>&1
