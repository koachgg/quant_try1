
# filepath: c:/Users/abhig/Documents/Quant/monitor_progress.py
import json
import time
import os
from datetime import datetime

def monitor_training():
    log_file = r'c:/Users/abhig/Documents/Quant\training_log.json'
    
    print("Training Progress Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    last_log_count = 0
    
    try:
        while True:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                    
                    # Show new logs
                    if len(logs) > last_log_count:
                        for log in logs[last_log_count:]:
                            status_icon = {
                                "start": "[START]",
                                "info": "[INFO]", 
                                "success": "[SUCCESS]",
                                "error": "[ERROR]",
                                "complete": "[COMPLETE]"
                            }.get(log.get("status", "info"), "[INFO]")
                            
                            print(f"{status_icon} {log['message']}")
                        
                        last_log_count = len(logs)
                        
                        # Check if complete
                        if logs and logs[-1].get("status") in ["complete", "error"]:
                            print("\nTraining finished!")
                            break
                
                except json.JSONDecodeError:
                    pass
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    monitor_training()
