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

    # Detect port availability
    import socket
    host = "127.0.0.1"
    port = 8000

    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex((host, p)) == 0

    if is_port_in_use(port):
        print(f"[WARNING] Port {port} is already in use. Checking for next available port...")
        while is_port_in_use(port):
            port += 1
        print(f"[INFO] Using available port: {port}")

    print()
    print("=" * 56)
    print("  Game server is running!")
    print(f"  Open your browser at: http://{host}:{port}/")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 56)
    print()

    # Open browser in a separate thread after 1.5 seconds
    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    # Launch Django development server with monitoring
    while True:
        try:
            res = subprocess.run([sys.executable, str(MANAGE_PY), "runserver", f"{host}:{port}"])
            if res.returncode == 0:
                break
            print(f"\n[ALERT] Server process exited with code {res.returncode}. Restarting in 2s (Ctrl+C to quit)...")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n[INFO] Server stopped by user.")
            break


if __name__ == "__main__":
    main()
