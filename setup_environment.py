import subprocess
import sys
import os
import time
import json
from datetime import datetime

class EnvironmentManager:
    """Manage virtual environment and long-running processes"""
    
    def __init__(self, project_path="c:/Users/abhig/Documents/Quant"):
        self.project_path = project_path
        self.venv_path = os.path.join(project_path, "kaggle_venv")
        self.log_file = os.path.join(project_path, "training_log.json")
        
    def create_venv(self):
        """Create and setup virtual environment"""
        print("🔧 Setting up virtual environment...")
        
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", self.venv_path], check=True)
        
        # Get activation script path
        if os.name == 'nt':  # Windows
            activate_script = os.path.join(self.venv_path, "Scripts", "activate.bat")
            pip_path = os.path.join(self.venv_path, "Scripts", "pip.exe")
        else:  # Unix/Linux
            activate_script = os.path.join(self.venv_path, "bin", "activate")
            pip_path = os.path.join(self.venv_path, "bin", "pip")
        
        # Install requirements
        print("📦 Installing requirements...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Virtual environment ready!")
        return activate_script
    
    def create_background_runner(self):
        """Create script to run training in background"""
        runner_script = f"""
# filepath: {self.project_path}/run_training.py
import sys
import os
import json
import time
import traceback
from datetime import datetime

# Add project path to Python path
sys.path.append(r'{self.project_path}')

def log_progress(message, status="info"):
    log_entry = {{ 
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "status": status
    }}
    
    log_file = r'{self.log_file}'
    
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
    
    print(f"[{{log_entry['timestamp']}}] {{message}}")

def main():
    try:
        log_progress("Starting background training process", "start")
        
        # Import and run strategy
        from kaggle_strategy import KaggleCompetitionStrategy
        
        strategy = KaggleCompetitionStrategy()
        
        log_progress("Starting development phase...")
        best_dev = strategy.development_phase()
        log_progress(f"Development complete. Best MSE: {{best_dev['results']['mse']:.8f}}")
        
        log_progress("Starting production phase...")
        predictions, predictor = strategy.production_phase(best_dev)
        log_progress("Production training complete")
        
        log_progress("Creating submission...")
        import pandas as pd
        test_df = pd.read_parquet('test.parquet')
        submission = strategy.create_submission(test_df, predictions)
        
        if strategy.validate_submission(submission):
            timestamp = int(time.time())
            filename = f'final_submission_{{timestamp}}.csv'
            submission.to_csv(filename, index=False)
            log_progress(f"Submission saved: {{filename}}", "success")
        else:
            log_progress("Submission validation failed", "error")
        
        log_progress("Training completed successfully!", "complete")
        
    except Exception as e:
        error_msg = f"Error during training: {{str(e)}}"
        log_progress(error_msg, "error")
        log_progress(traceback.format_exc(), "error")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
"""
        
        with open(os.path.join(self.project_path, "run_training.py"), 'w', encoding='utf-8') as f:
            f.write(runner_script)
        
        print("Background runner script created")
    
    def create_progress_monitor(self):
        """Create script to monitor training progress"""
        monitor_script = f"""
# filepath: {self.project_path}/monitor_progress.py
import json
import time
import os
from datetime import datetime

def monitor_training():
    log_file = r'{self.log_file}'
    
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
                            status_icon = {{
                                "start": "[START]",
                                "info": "[INFO]", 
                                "success": "[SUCCESS]",
                                "error": "[ERROR]",
                                "complete": "[COMPLETE]"
                            }}.get(log.get("status", "info"), "[INFO]")
                            
                            print(f"{{status_icon}} {{log['message']}}")
                        
                        last_log_count = len(logs)
                        
                        # Check if complete
                        if logs and logs[-1].get("status") in ["complete", "error"]:
                            print("\\nTraining finished!")
                            break
                
                except json.JSONDecodeError:
                    pass
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\\nMonitoring stopped")

if __name__ == "__main__":
    monitor_training()
"""
        
        with open(os.path.join(self.project_path, "monitor_progress.py"), 'w', encoding='utf-8') as f:
            f.write(monitor_script)
        
        print("Progress monitor script created")
    
    def create_batch_files(self):
        """Create convenient batch files for Windows"""
        
        # Start training batch file
        start_training_bat = f"""@echo off
cd /d "{self.project_path}"
call kaggle_venv\\Scripts\\activate.bat
echo Starting background training...
echo You can close this window and training will continue
start /b python run_training.py
echo Training started in background
echo Run monitor_progress.bat to check progress
pause
"""
        
        with open(os.path.join(self.project_path, "start_training.bat"), 'w', encoding='utf-8') as f:
            f.write(start_training_bat)
        
        # Monitor progress batch file
        monitor_bat = f"""@echo off
cd /d "{self.project_path}"
call kaggle_venv\\Scripts\\activate.bat
python monitor_progress.py
pause
"""
        
        with open(os.path.join(self.project_path, "monitor_progress.bat"), 'w', encoding='utf-8') as f:
            f.write(monitor_bat)
        
        # Quick setup batch file
        setup_bat = f"""@echo off
cd /d "{self.project_path}"
echo Setting up environment...
python setup_environment.py
echo Setup complete!
echo You can now run start_training.bat
pause
"""
        
        with open(os.path.join(self.project_path, "setup.bat"), 'w', encoding='utf-8') as f:
            f.write(setup_bat)
        
        print("Batch files created")

def main():
    """Setup complete environment"""
    print("KAGGLE COMPETITION ENVIRONMENT SETUP")
    print("=" * 50)
    
    manager = EnvironmentManager()
    
    # Create virtual environment
    activate_script = manager.create_venv()
    
    # Create helper scripts
    manager.create_background_runner()
    manager.create_progress_monitor()
    manager.create_batch_files()
    
    print("\\nEnvironment setup complete!")
    print("\\nNext steps:")
    print("1. Run 'setup.bat' to complete setup")
    print("2. Run 'start_training.bat' to start training")
    print("3. Run 'monitor_progress.bat' to check progress")
    print("\\nTraining runs in background - you can close windows safely!")

if __name__ == "__main__":
    main()
