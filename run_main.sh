set -e

bash scripts/etth1_plugin.sh "$@"
python scripts/summarize_etth1_96.py
