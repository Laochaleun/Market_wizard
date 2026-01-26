#!/bin/bash
# Market Wizard - start helper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="all"
FOREGROUND=0

if [ "$1" = "--foreground" ]; then
    FOREGROUND=1
    shift
fi

if [ -n "${1:-}" ]; then
    MODE="$1"
fi
PID_DIR="/tmp/market_wizard_pids"
LOG_DIR="/tmp/market_wizard_logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

start_service() {
    local name="$1"
    local cmd="$2"
    local log="$3"
    local pid_file="$PID_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "✅ $name already running (pid $pid)"
            return 0
        fi
        rm -f "$pid_file"
    fi

    echo "▶️ Starting $name..."
    if command -v setsid >/dev/null 2>&1; then
        setsid bash -c "$cmd" >"$log" 2>&1 &
    else
        nohup bash -c "$cmd" >"$log" 2>&1 &
    fi
    local pid=$!
    echo "$pid" >"$pid_file"
    echo "✅ $name started (pid $pid). Logs: $log"
}

case "$MODE" in
    gradio)
        if [ "$FOREGROUND" -eq 1 ]; then
            echo "Open: http://localhost:7860"
            ./run.sh gradio
        else
            start_service "gradio" "./run.sh gradio" "$LOG_DIR/gradio.log"
            echo "Open: http://localhost:7860"
        fi
        ;;
    api)
        if [ "$FOREGROUND" -eq 1 ]; then
            echo "Docs: http://localhost:8000/docs"
            ./run.sh api
        else
            start_service "api" "./run.sh api" "$LOG_DIR/api.log"
            echo "Docs: http://localhost:8000/docs"
        fi
        ;;
    all)
        if [ "$FOREGROUND" -eq 1 ]; then
            start_service "api" "./run.sh api" "$LOG_DIR/api.log"
            echo "Open: http://localhost:7860"
            echo "Docs: http://localhost:8000/docs"
            ./run.sh gradio
        else
            start_service "api" "./run.sh api" "$LOG_DIR/api.log"
            start_service "gradio" "./run.sh gradio" "$LOG_DIR/gradio.log"
            echo "Open: http://localhost:7860"
            echo "Docs: http://localhost:8000/docs"
        fi
        ;;
    *)
        echo "Usage: ./start.sh [--foreground] [gradio|api|all]"
        exit 1
        ;;
esac
