#!/usr/bin/env python3
import subprocess
import time
import sys
import signal
import threading
import webbrowser

# Terminal colors for better visibility
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color):
    print(f"{color}{text}{Colors.ENDC}")

def run_command(command, name, color):   
    print_colored(f"Starting {name}...", color)
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        for line in process.stdout:
            print_colored(f"[{name}] {line.strip()}", color)
        process.wait()
        print_colored(f"{name} process ended with code {process.returncode}", color)
        return process
    except Exception as e:
        print_colored(f"Error starting {name}: {e}", Colors.RED)
        return None

processes = []

def signal_handler(sig, frame):
    print_colored("\nShutting down all services...", Colors.YELLOW)
    for process in processes:
        if process and process.poll() is None:
            process.terminate()
    print_colored("All services stopped.", Colors.GREEN)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    try:
        subprocess.run(["rasa", "--version"], check=True, stdout=subprocess.PIPE)
    except Exception:
        print_colored("Rasa is not installed or not in PATH. Please install Rasa first.", Colors.RED)
        sys.exit(1)
    
    try:
        subprocess.run(["flask", "--version"], check=True, stdout=subprocess.PIPE)
    except Exception:
        print_colored("Flask is not installed. Installing it now...", Colors.YELLOW)
        subprocess.run([sys.executable, "-m", "pip", "install", "flask"])
    
    action_server = threading.Thread(
        target=run_command,
        args=("rasa run actions", "Action Server", Colors.BLUE)
    )
    action_server.daemon = True
    action_server.start()
    time.sleep(2)
    
    rasa_server = threading.Thread(
        target=run_command,
        args=("rasa run --enable-api --cors '*'", "Rasa Server", Colors.GREEN)
    )
    rasa_server.daemon = True
    rasa_server.start()
    time.sleep(5)
    
    flask_url = "http://localhost:8501"
    print_colored("Starting Flask frontend...", Colors.YELLOW)
    print_colored(f"Flask app will be available at: {flask_url}", Colors.YELLOW) 
    threading.Timer(2, lambda: webbrowser.open(flask_url)).start()
    
    flask_process = run_command("python app.py", "Flask", Colors.YELLOW)
    processes.append(flask_process)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    print_colored("=" * 50, Colors.BOLD)
    print_colored("  Starting Banking Virtual Assistant", Colors.BOLD)
    print_colored("=" * 50, Colors.BOLD)
    main()
