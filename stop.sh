#!/bin/bash
# Market Wizard - stop helper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE=${1:-all}
PID_DIR="/tmp/market_wizard_pids"

stop_service() {
    local name="$1"
    local port="$2"
    local pid_file="$PID_DIR/${name}.pid"
    local pid=""

    if [ ! -f "$pid_file" ]; then
        if [ -n "$port" ]; then
            pid=$(lsof -ti :"$port" 2>/dev/null | head -n 1)
        fi
        if [ -z "$pid" ]; then
            echo "ℹ️  $name not running (no pid file)"
            return 0
        fi
    else
        pid=$(cat "$pid_file")
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ℹ️  $name not running (stale pid $pid)"
        rm -f "$pid_file"
        return 0
    fi

    echo "⏹ Stopping $name (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    kill -TERM -"$pid" 2>/dev/null || true

    for _ in {1..10}; do
        if kill -0 "$pid" 2>/dev/null; then
            sleep 0.5
        else
            break
        fi
    done

    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
        kill -KILL -"$pid" 2>/dev/null || true
    fi

    pkill -P "$pid" 2>/dev/null || true
    rm -f "$pid_file"
    echo "✅ $name stopped"
}

stop_by_port() {
    local port="$1"
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null)
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "⏹ Stopping processes on port $port..."
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 0.5
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

case "$MODE" in
    gradio)
        stop_service "gradio" "7860"
        stop_by_port "7860"
        ;;
    api)
        stop_service "api" "8000"
        stop_by_port "8000"
        ;;
    all)
        stop_service "gradio" "7860"
        stop_service "api" "8000"
        stop_by_port "7860"
        stop_by_port "8000"
        ;;
    *)
        echo "Usage: ./stop.sh [gradio|api|all]"
        exit 1
        ;;
esac
