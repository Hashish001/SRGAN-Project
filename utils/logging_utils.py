import os
import json

def save_upload_log(log_data):
    """Save upload log to JSON file"""
    log_file = "uploads/logs/upload_history.json"
    
    # Load existing logs
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Add new log
    logs.append(log_data)
    
    # Save logs
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)