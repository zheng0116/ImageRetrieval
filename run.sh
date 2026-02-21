#!/usr/bin/env bash
set -euo pipefail

case ${1:-} in
"start")
    python3 main.py
    ;;
"install")
    python3 -m venv .venv
    source ./.venv/bin/activate
    pip3 install -r requirements.txt
    ;;
*)
    echo "Usage: sh run.sh {start|install}"
    exit 1
    ;;
esac
