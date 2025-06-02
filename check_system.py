import psutil
import platform
import sys
import subprocess

def check_system_specs():
    """Check if laptop specs are sufficient for training"""
    print("SYSTEM SPECIFICATION CHECK")
    print("=" * 50)
    
    # CPU Information
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    print(f"CPU Cores: {cpu_count}")
    if cpu_freq:
        print(f"CPU Frequency: {cpu_freq.current:.0f} MHz")
    
    # Memory Information
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    print(f"Total RAM: {memory_gb:.1f} GB")
    print(f"Available RAM: {available_gb:.1f} GB")
    
    # Storage Information
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    print(f"Free Storage: {free_gb:.1f} GB")
    
    # Python Version
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 30)
    
    if memory_gb >= 8:
        print("✅ RAM: Sufficient (8GB+ recommended)")
    elif memory_gb >= 4:
        print("⚠️  RAM: Minimum (4-8GB) - use smaller subset")
    else:
        print("❌ RAM: Insufficient (<4GB)")
    
    if cpu_count >= 4:
        print("✅ CPU: Good for parallel processing")
    else:
        print("⚠️  CPU: Limited cores - training will be slower")
    
    if free_gb >= 5:
        print("✅ Storage: Sufficient")
    else:
        print("❌ Storage: Need at least 5GB free")
    
    # Time estimates based on specs
    print(f"\nESTIMATED TRAINING TIMES:")
    print("-" * 30)
    
    if memory_gb >= 16 and cpu_count >= 8:
        print("Development Phase: 15-30 minutes")
        print("Production Phase: 1-2 hours")
        print("Grade: EXCELLENT 🚀")
    elif memory_gb >= 8 and cpu_count >= 4:
        print("Development Phase: 30-60 minutes") 
        print("Production Phase: 2-4 hours")
        print("Grade: GOOD ✅")
    else:
        print("Development Phase: 60-120 minutes")
        print("Production Phase: 4-8 hours")
        print("Grade: WORKABLE ⚠️")
        print("Recommendation: Use cloud computing (Kaggle, Colab)")

if __name__ == "__main__":
    check_system_specs()
