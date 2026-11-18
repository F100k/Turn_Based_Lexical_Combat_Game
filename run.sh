#!/usr/bin/env bash
# Turn-Based Lexical Combat Game - Monitored Launcher

# Change directory to the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "  Turn-Based Lexical Combat Game - Launcher & Monitor"
echo "========================================================"
echo ""

# Locate Python
if command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
elif command -v python &>/dev/null; then
    PYTHON_EXE="python"
else
    echo "[ERROR] Python was not found in your PATH!"
    echo "Please install Python 3.10+."
    exit 1
fi

# Detect or create virtual environment
VENV_PATH=""
if [ -f "venv/bin/activate" ]; then
    VENV_PATH="venv"
elif [ -f ".venv/bin/activate" ]; then
    VENV_PATH=".venv"
else
    echo "[INFO] No virtual environment detected. Creating 'venv'..."
    if $PYTHON_EXE -m venv venv 2>/dev/null; then
        VENV_PATH="venv"
        source "$VENV_PATH/bin/activate"
        echo "[INFO] Installing dependencies from requirements.txt..."
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "[WARNING] Could not create virtual environment. Using system Python..."
    fi
fi

if [ -n "$VENV_PATH" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "[INFO] Activating virtual environment ($VENV_PATH)..."
    source "$VENV_PATH/bin/activate"
fi

# Ensure dependencies are present
if ! python -c "import django" &>/dev/null; then
    echo "[INFO] Django not installed. Installing requirements..."
    pip install -r requirements.txt
fi

# Run migrations
echo "[INFO] Applying database migrations..."
python dev/manage.py migrate

HOST="127.0.0.1"
PORT=8000

# Helper to check if a port is in use
is_port_in_use() {
    local check_port="$1"
    python -c "
import socket, sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(0.3)
    sys.exit(0 if s.connect_ex(('$HOST', int('$check_port'))) == 0 else 1)
" 2>/dev/null
}

# Helper to release occupied port
free_port() {
    local target_port="$1"
    if is_port_in_use "$target_port"; then
        echo "[WARNING] Port $target_port is currently in use."
        echo "[INFO] Terminating stale game server processes on port $target_port..."
        pkill -f "dev/manage.py runserver.*$target_port" 2>/dev/null || true
        if command -v fuser &>/dev/null; then
            fuser -k -n tcp "$target_port" 2>/dev/null || true
        elif command -v lsof &>/dev/null; then
            local pids
            pids=$(lsof -ti:"$target_port" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                kill -9 $pids 2>/dev/null || true
            fi
        fi
        sleep 1
    fi
}

# Free port 8000 if a previous run is lingering
free_port "$PORT"

# If still busy (e.g. permission or another service), find next available port
while is_port_in_use "$PORT"; do
    echo "[WARNING] Port $PORT could not be released. Trying port $((PORT + 1))..."
    PORT=$((PORT + 1))
done

# Display info
echo ""
echo "========================================================"
echo "  Turn-Based Lexical Combat Game is running!"
echo "  URL: http://$HOST:$PORT/"
echo "  Process Monitoring: ACTIVE (Auto-restart enabled)"
echo "  Press Ctrl+C to stop the server."
echo "========================================================"
echo ""

SERVER_PID=""
SHUTDOWN_REQUESTED=0

cleanup() {
    SHUTDOWN_REQUESTED=1
    echo ""
    echo "[INFO] Stopping game server..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Also ensure any child processes on this port are closed
    pkill -P $$ 2>/dev/null || true
    echo "[INFO] Server stopped cleanly. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Open browser once on initial launch
if command -v xdg-open &>/dev/null; then
    (sleep 2 && xdg-open "http://$HOST:$PORT/" 2>/dev/null) &
elif command -v open &>/dev/null; then
    (sleep 2 && open "http://$HOST:$PORT/" 2>/dev/null) &
fi

# Continuous process supervisor loop
while [ "$SHUTDOWN_REQUESTED" -eq 0 ]; do
    python dev/manage.py runserver "$HOST:$PORT" &
    SERVER_PID=$!

    # Wait for the server process to exit
    wait "$SERVER_PID" 2>/dev/null
    EXIT_CODE=$?
    SERVER_PID=""

    if [ "$SHUTDOWN_REQUESTED" -eq 1 ]; then
        break
    fi

    # If exit code was not normal (0) or manual interrupt (130)
    if [ "$EXIT_CODE" -ne 0 ] && [ "$EXIT_CODE" -ne 130 ]; then
        echo ""
        echo "[ALERT] Game server exited unexpectedly (Exit Code: $EXIT_CODE)."
        echo "[INFO] Checking port $PORT and restarting server in 2 seconds..."
        free_port "$PORT"
        sleep 2
    else
        echo "[INFO] Server process terminated normally."
        break
    fi
done
