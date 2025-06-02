import pandas as pd
import numpy as np
from fast_pipeline import FastIVPredictor
import time
import gc

class KaggleCompetitionStrategy:
    """Complete strategy for winning Kaggle competitions"""
    
    def __init__(self):
        self.experiments = []
        self.best_model = None
        self.submission_count = 0
        self.max_daily_submissions = 5
        
    def development_phase(self):
        """Phase 1: Rapid experimentation on subset (30-60 minutes total)"""
        print("DEVELOPMENT PHASE - Rapid Experimentation")
        print("=" * 60)
        
        # Load data
        train_df = pd.read_parquet('train.parquet')
        test_df = pd.read_parquet('test.parquet')
        
        experiments = [
            {"subset_ratio": 0.05, "max_features": 30, "name": "ultra_fast"},
            {"subset_ratio": 0.1, "max_features": 50, "name": "balanced"},
            {"subset_ratio": 0.2, "max_features": 100, "name": "thorough"}
        ]
        
        best_experiment = None
        best_score = float('inf')
        
        for i, exp_config in enumerate(experiments):
            print(f"\nExperiment {i+1}/3: {exp_config['name']}")
            print("-" * 40)
            
            # Create predictor
            predictor = FastIVPredictor(
                use_subset=True,
                subset_ratio=exp_config['subset_ratio'],
                max_features=exp_config['max_features']
            )
            
            # Run development
            results = predictor.rapid_development(train_df)
            
            if results and results['mse'] < best_score:
                best_score = results['mse']
                best_experiment = {
                    'config': exp_config,
                    'results': results,
                    'predictor': predictor
                }
            
            # Estimate full training time
            estimated_full_time = results['time'] / exp_config['subset_ratio']
            print(f"Estimated full training time: {estimated_full_time/60:.1f} minutes")
            
            # Memory cleanup
            del predictor
            gc.collect()
        
        print(f"\nBest Development Model:")
        print(f"   MSE: {best_experiment['results']['mse']:.8f}")
        print(f"   Config: {best_experiment['config']['name']}")
        
        return best_experiment
    
    def production_phase(self, best_dev_config):
        """Phase 2: Production training on full data (2-4 hours)"""
        print("\nPRODUCTION PHASE - Full Data Training")
        print("=" * 60)
        
        # Load full data
        train_df = pd.read_parquet('train.parquet')
        test_df = pd.read_parquet('test.parquet')
        
        # Create production predictor with best config
        predictor = FastIVPredictor(
            use_subset=False,  # Use full data
            max_features=best_dev_config['config']['max_features']
        )
        
        # Scale to production
        production_models = predictor.scale_to_production(
            train_df, best_dev_config['results']
        )
        
        # Make predictions
        predictions = self.make_production_predictions(
            test_df, production_models, predictor
        )
        
        return predictions, predictor
    
    def make_production_predictions(self, test_df, models, predictor):
        """Make final predictions for submission"""
        print("\nMaking Production Predictions...")
        
        # Prepare test data
        target_cols = list(models.keys())
        X_test, feature_cols = predictor.prepare_data(test_df, target_cols)
        
        predictions = {}
        
        for target_col, model in models.items():
            pred = model.predict(X_test)
            pred = np.clip(pred, 0.005, 3.0)  # Apply bounds
            predictions[target_col] = pred
            print(f"  {target_col}: μ={np.mean(pred):.6f}")
        
        return predictions
    
    def create_submission(self, test_df, predictions):
        """Create competition submission"""
        print("\nCreating Submission...")
        
        submission = pd.DataFrame({'timestamp': test_df['timestamp']})
        
        # Add all required IV columns
        all_strikes = {
            'call': list(range(24000, 26600, 100)),
            'put': list(range(23000, 25600, 100))
        }
        
        for option_type, strikes in all_strikes.items():
            for strike in strikes:
                col_name = f'{option_type}_iv_{strike}'
                
                if col_name in predictions:
                    submission[col_name] = predictions[col_name]
                elif col_name in test_df.columns:
                    # Keep original non-NaN values
                    submission[col_name] = test_df[col_name].fillna(0.15)
                else:
                    # Conservative fallback
                    submission[col_name] = 0.15
        
        # Final cleanup
        iv_cols = [col for col in submission.columns if '_iv_' in col]
        for col in iv_cols:
            submission[col] = np.clip(submission[col], 0.005, 3.0)
            submission[col] = submission[col].fillna(0.15)
        
        return submission
    
    def validate_submission(self, submission):
        """Validate submission before upload"""
        print("\nValidating Submission...")
        
        # Check shape
        expected_cols = ['timestamp'] + [f'call_iv_{s}' for s in range(24000, 26600, 100)] + \
                       [f'put_iv_{s}' for s in range(23000, 25600, 100)]
        
        if list(submission.columns) != expected_cols:
            print("Column mismatch!")
            return False
        
        # Check for NaN values
        if submission.isnull().any().any():
            print("NaN values found!")
            return False
        
        # Check value ranges
        iv_cols = [col for col in submission.columns if '_iv_' in col]
        for col in iv_cols:
            if submission[col].min() < 0.001 or submission[col].max() > 10:
                print(f"Unrealistic values in {col}!")
                return False
        
        print("Submission validation passed!")
        return True
    
    def execute_full_strategy(self):
        """Execute complete competition strategy"""
        print("KAGGLE COMPETITION STRATEGY")
        print("Goal: Top 10 Performance")
        print("Time Budget: 4-6 hours total")
        print("=" * 60)
        
        total_start = time.time()
        
        # Phase 1: Development (30-60 minutes)
        best_dev = self.development_phase()
        
        # Phase 2: Production (2-4 hours)
        predictions, predictor = self.production_phase(best_dev)
        
        # Phase 3: Submission
        test_df = pd.read_parquet('test.parquet')
        submission = self.create_submission(test_df, predictions)
        
        # Validate and save
        if self.validate_submission(submission):
            timestamp = int(time.time())
            filename = f'submission_strategy_{timestamp}.csv'
            submission.to_csv(filename, index=False)
            print(f"💾 Submission saved: {filename}")
        
        total_time = time.time() - total_start
        print(f"\nStrategy Complete!")
        print(f"Total Time: {total_time/3600:.1f} hours")
        print(f"Ready for top 10 performance!")
        
        return submission

def main():
    """Execute the complete winning strategy"""
    strategy = KaggleCompetitionStrategy()
    submission = strategy.execute_full_strategy()
    return submission

if __name__ == "__main__":
    main()
