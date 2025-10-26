#!/usr/bin/env python3
"""
Start TensorBoard server for monitoring training
"""

import os
import sys
import subprocess
import webbrowser
from datetime import datetime

def start_tensorboard(logdir="logs", port=6006):
    """
    Start TensorBoard server
    """
    print("=" * 60)
    print("TENSORBOARD LAUNCHER")
    print("=" * 60)
    print(f"Log directory: {logdir}")
    print(f"Port: {port}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Check if log directory exists
    if not os.path.exists(logdir):
        print(f"⚠️  Log directory '{logdir}' not found.")
        print(f"   It will be created when training starts.")
        print(f"   Start training first, then run TensorBoard again.")
        return

    # Count log files
    try:
        log_files = os.listdir(logdir)
        if log_files:
            print(f"Found {len(log_files)} log file(s)")
        else:
            print("Log directory is empty - Start training first")
    except:
        pass

    print("\nStarting TensorBoard...")
    print(f"URL: http://localhost:{port}")
    print("\nPress Ctrl+C to stop TensorBoard")
    print("-" * 60)

    # Open browser after a short delay
    import threading
    def open_browser():
        import time
        time.sleep(3)
        webbrowser.open(f"http://localhost:{port}")

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start TensorBoard
    try:
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", logdir,
            "--port", str(port),
            "--host", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ TensorBoard stopped")
    except Exception as e:
        print(f"\n❌ Error starting TensorBoard: {e}")
        print("\nTry installing tensorboard:")
        print("  pip install tensorboard")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Launch TensorBoard for monitoring')
    parser.add_argument('--logdir', default='logs', help='Log directory (default: logs)')
    parser.add_argument('--port', type=int, default=6006, help='Port number (default: 6006)')

    args = parser.parse_args()

    start_tensorboard(args.logdir, args.port)