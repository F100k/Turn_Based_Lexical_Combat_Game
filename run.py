#!/usr/bin/env python3
"""Cross-platform launcher for Turn-Based Lexical Combat Game."""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MANAGE_PY = ROOT_DIR / "dev" / "manage.py"


def main():
    print("=" * 56)
    print("  Turn-Based Lexical Combat Game - Cross-Platform Launcher")
    print("=" * 56)
    print()

    if not MANAGE_PY.exists():
        print(f"[ERROR] Could not find manage.py at {MANAGE_PY}")
        sys.exit(1)

    # Check for Django
    try:
        import django
    except ImportError:
        print("[INFO] Django is not installed in the current Python environment.")
        req_file = ROOT_DIR / "requirements.txt"
        if req_file.exists():
            print("[INFO] Installing dependencies from requirements.txt...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)
        else:
            print("[ERROR] requirements.txt not found. Please install django manually.")
            sys.exit(1)

    # Apply database migrations
    print("[INFO] Applying database migrations...")
    subprocess.run([sys.executable, str(MANAGE_PY), "migrate"], check=True)

    print()
    print("=" * 56)
    print("  Game server is running!")
    print("  Open your browser at: http://127.0.0.1:8000/")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 56)
    print()

    # Open browser in a separate thread after 1.5 seconds
    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open("http://127.0.0.1:8000/")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    # Launch Django development server
    try:
        subprocess.run([sys.executable, str(MANAGE_PY), "runserver", "127.0.0.1:8000"])
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user.")


if __name__ == "__main__":
    main()
