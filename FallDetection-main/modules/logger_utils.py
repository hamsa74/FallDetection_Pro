import datetime

def log_event(status):
    with open("output/logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Status: {status}\n")
    print(f"Log updated: {status}")