#!/usr/bin/env python3
"""
Simple real-time resource monitor for terminal
"""

import os
import time
import psutil
from datetime import datetime

def monitor_resources(interval=2):
    """Monitor system resources in real-time"""

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')

            # Get resource info
            cpu = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Find Python processes
            python_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        if proc.info['cpu_percent'] > 0:  # Only show active processes
                            python_procs.append(proc.info)
                except:
                    pass

            # Display
            print(f"Resource Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 50)
            print(f"System CPU: {cpu:>6.1f}%")
            print(f"Memory:     {memory.percent:>6.1f}% ({memory.used/1024/1024/1024:6.1f} GB / {memory.total/1024/1024/1024:.1f} GB)")
            print(f"Disk Free:  {disk.free/1024/1024/1024:>6.1f} GB")

            if python_procs:
                print("\nPython Processes:")
                print("-" * 50)
                for proc in sorted(python_procs, key=lambda x: x['cpu_percent'], reverse=True):
                    print(f"PID {proc['pid']:<6} CPU: {proc['cpu_percent']:>5.1f}%  Memory: {proc['memory_percent']:>5.1f}%")

            print(f"\nUpdating every {interval} seconds (Press Ctrl+C to exit)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simple resource monitor')
    parser.add_argument('--interval', type=int, default=2, help='Update interval in seconds')
    args = parser.parse_args()

    monitor_resources(args.interval)