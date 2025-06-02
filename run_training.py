
# filepath: c:/Users/abhig/Documents/Quant/run_training.py
import sys
import os
import json
import time
import traceback
from datetime import datetime

# Add project path to Python path
sys.path.append(r'c:/Users/abhig/Documents/Quant')

def log_progress(message, status="info"):
    log_entry = { 
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "status": status
    }
    
    log_file = r'c:/Users/abhig/Documents/Quant\training_log.json'
    
    # Read existing logs
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
    
    # Add new log
    logs.append(log_entry)
    
    # Keep only last 100 logs
    logs = logs[-100:]
    
    # Write back
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"[{log_entry['timestamp']}] {message}")

def main():
    try:
        log_progress("Starting background training process", "start")
        
        # Import and run strategy
        from kaggle_strategy import KaggleCompetitionStrategy
        
        strategy = KaggleCompetitionStrategy()
        
        log_progress("Starting development phase...")
        best_dev = strategy.development_phase()
        log_progress(f"Development complete. Best MSE: {best_dev['results']['mse']:.8f}")
        
        log_progress("Starting production phase...")
        predictions, predictor = strategy.production_phase(best_dev)
        log_progress("Production training complete")
        
        log_progress("Creating submission...")
        import pandas as pd
        test_df = pd.read_parquet('test.parquet')
        submission = strategy.create_submission(test_df, predictions)
        
        if strategy.validate_submission(submission):
            timestamp = int(time.time())
            filename = f'final_submission_{timestamp}.csv'
            submission.to_csv(filename, index=False)
            log_progress(f"Submission saved: {filename}", "success")
        else:
            log_progress("Submission validation failed", "error")
        
        log_progress("Training completed successfully!", "complete")
        
    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        log_progress(error_msg, "error")
        log_progress(traceback.format_exc(), "error")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
