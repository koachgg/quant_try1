import subprocess
import sys
import os

def fix_environment():
    """Fix the virtual environment and install missing packages"""
    print("FIXING ENVIRONMENT")
    print("=" * 30)
    
    project_path = "c:/Users/abhig/Documents/Quant"
    venv_path = os.path.join(project_path, "kaggle_venv")
    
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Check if virtual environment exists
    if not os.path.exists(pip_path):
        print("Virtual environment not found. Creating...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    
    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install packages optimized for your specs (16 cores, 16GB RAM)
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.6.0",
        "optuna>=3.0.0",
        "pyarrow>=8.0.0",
        "scipy>=1.9.0",
        "psutil>=5.9.0",
        "tqdm>=4.64.0"
    ]
    
    print("Installing packages optimized for your specs...")
    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([pip_path, "install", package], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {package} installation timed out")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
    
    # Test imports
    print("\nTesting imports...")
    test_script = f'''
import sys
sys.path.append(r"{project_path}")
try:
    import pandas as pd
    print("✅ pandas working")
    import numpy as np
    print("✅ numpy working")
    import lightgbm as lgb
    print("✅ lightgbm working")
    import xgboost as xgb
    print("✅ xgboost working")
    import sklearn
    print("✅ sklearn working")
    print("🚀 All packages ready!")
except Exception as e:
    print(f"❌ Import error: {{e}}")
'''
    
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    subprocess.run([python_path, "test_imports.py"])
    os.remove("test_imports.py")
    
    print("\n✅ Environment fix complete!")

if __name__ == "__main__":
    fix_environment()
