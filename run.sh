#!/usr/bin/env bash
set -e

# Change directory to the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "  Turn-Based Lexical Combat Game - Launcher"
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

# Display info
echo ""
echo "========================================================"
echo "  Turn-Based Lexical Combat Game is running!"
echo "  Open your browser at: http://127.0.0.1:8000/"
echo "  Press Ctrl+C to stop the server."
echo "========================================================"
echo ""

# Try to open the browser automatically in desktop environments
if command -v xdg-open &>/dev/null; then
    (sleep 2 && xdg-open "http://127.0.0.1:8000/" 2>/dev/null) &
elif command -v open &>/dev/null; then
    (sleep 2 && open "http://127.0.0.1:8000/" 2>/dev/null) &
fi

exec python dev/manage.py runserver 127.0.0.1:8000
