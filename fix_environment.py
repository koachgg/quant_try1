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
    
    # Install packages one by one to catch errors
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.6.0",
        "optuna>=3.0.0",
        "pyarrow>=8.0.0",
        "scipy>=1.9.0",
        "psutil>=5.9.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([pip_path, "install", package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                failed_packages.append(package)
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed packages: {failed_packages}")
        print("Trying alternative installation...")
        
        # Try installing without version constraints
        simple_packages = [pkg.split('>=')[0] for pkg in failed_packages]
        for package in simple_packages:
            try:
                subprocess.run([pip_path, "install", package], check=True)
                print(f"✅ {package} installed (simple version)")
            except:
                print(f"❌ Still failed: {package}")
    
    print("\n✅ Environment fix complete!")

if __name__ == "__main__":
    fix_environment()
