from datetime import datetime
import os

def log_event(message):
    if not os.path.exists('output'):
        os.makedirs('output')
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('output/logs.txt', 'a') as f:
        f.write(f"[{timestamp}] {message}\n")