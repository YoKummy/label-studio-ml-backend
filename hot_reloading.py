import threading
import time
import json

CONFIG_FILE = "config.json"

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def printer_loop():
    while True:
        config = load_config()  # read the latest config each time
        print(int(config["value"]))
        time.sleep(1)

# Run printer in a separate thread
t = threading.Thread(target=printer_loop, daemon=True)
t.start()

# Keep main thread alive 
while True:
    time.sleep(1)
