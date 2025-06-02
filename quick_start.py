from kaggle_strategy import KaggleCompetitionStrategy
import time

def main():
    """Quick start for competition"""
    print("🚀 QUICK START - Kaggle Competition")
    print("⚡ Fast track to winning submission")
    print("=" * 50)
    
    start_time = time.time()
    
    # Execute strategy
    strategy = KaggleCompetitionStrategy()
    submission = strategy.execute_full_strategy()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Complete! Time: {elapsed/3600:.1f} hours")
    print("📤 Ready to submit to Kaggle!")

if __name__ == "__main__":
    main()
