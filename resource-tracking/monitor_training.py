#!/usr/bin/env python3
"""
Resource Monitoring Script for Training
Monitor CPU, Memory, and GPU usage during model training
"""

import os
import time
import psutil
import GPUtil
import json
from datetime import datetime

def get_system_info():
    """Get system resource information"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'memory': psutil.virtual_memory()._asdict(),
        'disk': psutil.disk_usage('/')._asdict(),
    }

    # Get GPU info if available
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            info['gpu'] = {
                'name': gpus[0].name,
                'load': gpus[0].load * 100,
                'memory_used': gpus[0].memoryUsed,
                'memory_total': gpus[0].memoryTotal,
                'temperature': gpus[0].temperature
            }
    except:
        info['gpu'] = None

    return info

def monitor_process(pid, log_file='training_monitor.log', interval=5):
    """Monitor a specific process by PID"""
    print(f"Starting monitoring for process {pid}")
    print(f"Logging to {log_file}")
    print(f"Interval: {interval} seconds")
    print("-" * 60)

    logs = []

    try:
        process = psutil.Process(pid)

        while True:
            # Get system info
            sys_info = get_system_info()

            # Get process-specific info
            try:
                process_info = {
                    'pid': pid,
                    'name': process.name(),
                    'cpu_percent': process.cpu_percent(),
                    'memory_info': process.memory_info()._asdict(),
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'status': process.status()
                }
            except psutil.NoSuchProcess:
                print(f"Process {pid} no longer exists")
                break

            # Combine info
            log_entry = {
                'system': sys_info,
                'process': process_info
            }
            logs.append(log_entry)

            # Display info
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print(f"Process: {process_info['name']} (PID: {process_info['pid']})")
            print(f"Status: {process_info['status']}")
            print(f"\nCPU Usage: {process_info['cpu_percent']:.1f}%")
            print(f"Memory Usage: {process_info['memory_percent']:.1f}% ({process_info['memory_info']['rss']/1024/1024:.1f} MB)")
            print(f"Threads: {process_info['num_threads']}")

            print(f"\nSystem CPU: {sys_info['cpu_percent']:.1f}%")
            print(f"System Memory: {sys_info['memory']['percent']:.1f}% ({sys_info['memory']['used']/1024/1024/1024:.1f} GB / {sys_info['memory']['total']/1024/1024/1024:.1f} GB)")
            print(f"Disk Free: {sys_info['disk']['free']/1024/1024/1024:.1f} GB")

            if sys_info['gpu']:
                print(f"\nGPU: {sys_info['gpu']['name']}")
                print(f"GPU Load: {sys_info['gpu']['load']:.1f}%")
                print(f"GPU Memory: {sys_info['gpu']['memory_used']} MB / {sys_info['gpu']['memory_total']} MB")
                if sys_info['gpu']['temperature']:
                    print(f"GPU Temperature: {sys_info['gpu']['temperature']}°C")

            # Log to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        print(f"Logs saved to {log_file}")

def find_python_process():
    """Find running Python training processes"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'Python' or 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'train.py' in cmdline:
                    python_processes.append((proc.info['pid'], cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_processes

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor training process resources')
    parser.add_argument('--pid', type=int, help='Process ID to monitor')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    parser.add_argument('--log', default='training_monitor.log', help='Log file name')
    parser.add_argument('--list', action='store_true', help='List Python training processes')

    args = parser.parse_args()

    if args.list:
        processes = find_python_process()
        if processes:
            print("Found Python training processes:")
            for pid, cmdline in processes:
                print(f"  PID: {pid} - {cmdline}")
        else:
            print("No Python training processes found")
    elif args.pid:
        monitor_process(args.pid, args.log, args.interval)
    else:
        print("Usage:")
        print("  1. Start training in one terminal: python train.py")
        print("  2. Find the PID: python monitor_training.py --list")
        print("  3. Monitor the process: python monitor_training.py --pid <PID>")
        print("\nOptions:")
        print("  --interval N    Monitor every N seconds (default: 5)")
        print("  --log FILE      Log to FILE (default: training_monitor.log)")