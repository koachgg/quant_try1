import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import gc
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class OptimizedIVPredictor:
    """Optimized predictor for 16-core, 16GB system"""
    
    def __init__(self, use_subset=True, subset_ratio=0.15):
        self.use_subset = use_subset
        self.subset_ratio = subset_ratio  # Higher ratio for your good specs
        self.scaler = RobustScaler()
        self.models = {}
        self.n_jobs = 14  # Leave 2 cores for system
        
    def create_optimized_subset(self, df):
        """Create larger subset optimized for your specs"""
        if not self.use_subset:
            return df
        
        # With 16GB RAM, we can handle larger subsets
        subset_size = max(5000, int(len(df) * self.subset_ratio))
        subset = df.sample(n=subset_size, random_state=42)
        print(f"Using optimized subset: {len(subset)} rows ({len(subset)/len(df)*100:.1f}%)")
        return subset
    
    def fast_feature_engineering(self, df):
        """Optimized feature engineering for your system"""
        print("Optimized feature engineering...")
        start_time = time.time()
        
        # Clean IV data aggressively
        iv_cols = [col for col in df.columns if '_iv_' in col]
        for col in iv_cols:
            if col in df.columns:
                # Remove extreme outliers
                df[col] = df[col].clip(0.001, 5.0)
                # Handle negative values
                df.loc[df[col] < 0, col] = np.nan
        
        # Enhanced volatility surface features
        if iv_cols:
            df['iv_mean'] = df[iv_cols].mean(axis=1, skipna=True)
            df['iv_median'] = df[iv_cols].median(axis=1, skipna=True)
            df['iv_std'] = df[iv_cols].std(axis=1, skipna=True)
            df['iv_range'] = df[iv_cols].max(axis=1) - df[iv_cols].min(axis=1)
            df['iv_skew'] = df[iv_cols].skew(axis=1, skipna=True)
            df['iv_count'] = df[iv_cols].count(axis=1)
        
        # Enhanced X features (use more with your specs)
        x_cols = [col for col in df.columns if col.startswith('X')][:25]  # More features
        if x_cols:
            df['X_mean'] = df[x_cols].mean(axis=1)
            df['X_median'] = df[x_cols].median(axis=1)
            df['X_std'] = df[x_cols].std(axis=1)
            df['X_skew'] = df[x_cols].skew(axis=1)
            df['X_range'] = df[x_cols].max(axis=1) - df[x_cols].min(axis=1)
            
            # Interactions for top X features
            top_x = x_cols[:8]
            for i in range(len(top_x)):
                for j in range(i+1, min(i+3, len(top_x))):
                    col1, col2 = top_x[i], top_x[j]
                    df[f'{col1}_{col2}_ratio'] = df[col1] / (np.abs(df[col2]) + 1e-8)
        
        # Comprehensive moneyness features
        if 'underlying' in df.columns:
            strikes = [24000, 24500, 25000, 25500, 26000]
            for strike in strikes:
                df[f'moneyness_{strike}'] = np.log(strike / df['underlying'])
                df[f'abs_moneyness_{strike}'] = np.abs(np.log(strike / df['underlying']))
                df[f'strike_distance_{strike}'] = np.abs(df['underlying'] - strike)
        
        print(f"Feature engineering: {time.time() - start_time:.1f}s")
        return df
    
    def optimized_model_selection(self, X, y):
        """Model selection optimized for your hardware"""
        print("Optimized model selection...")
        start_time = time.time()
        
        # Models optimized for 16 cores
        models = {
            'lgb_fast': lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, n_jobs=self.n_jobs, random_state=42, verbose=-1
            ),
            'lgb_deep': lgb.LGBMRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.03,
                num_leaves=63, n_jobs=self.n_jobs, random_state=42, verbose=-1
            ),
            'xgb_fast': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                n_jobs=self.n_jobs, random_state=42, verbosity=0
            ),
            'rf_parallel': RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=42, 
                n_jobs=self.n_jobs
            )
        }
        
        best_score = float('inf')
        best_model_name = None
        
        # Parallel cross-validation
        for name, model in models.items():
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                mse = mean_squared_error(y_val, pred)
                cv_scores.append(mse)
            
            avg_mse = np.mean(cv_scores)
            print(f"{name}: MSE = {avg_mse:.8f}")
            
            if avg_mse < best_score:
                best_score = avg_mse
                best_model_name = name
        
        print(f"Model selection: {time.time() - start_time:.1f}s")
        return best_model_name, models[best_model_name], best_score
    
    def rapid_development(self, train_df):
        """Optimized development cycle for your specs"""
        print("OPTIMIZED DEVELOPMENT CYCLE")
        print("Using 16 cores, 16GB RAM optimization")
        print("=" * 50)
        
        start_time = time.time()
        
        # Create optimized subset
        train_subset = self.create_optimized_subset(train_df)
        
        # Feature engineering
        train_features = self.fast_feature_engineering(train_subset)
        
        # Target selection
        target_cols = [col for col in train_features.columns if '_iv_' in col]
        print(f"Found {len(target_cols)} IV targets")
        
        # Feature selection
        exclude_cols = ['timestamp', 'expiry'] + target_cols
        feature_cols = [col for col in train_features.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} features")
        
        # Train on representative target
        main_target = 'call_iv_25000' if 'call_iv_25000' in target_cols else target_cols[0]
        mask = train_features[main_target].notna()
        
        if mask.sum() < 100:
            print(f"Insufficient data for {main_target}")
            return None
        
        X = train_features.loc[mask, feature_cols].fillna(0)
        y = train_features.loc[mask, main_target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Training on {len(X)} samples with {X_scaled.shape[1]} features")
        
        # Model selection
        best_model_name, best_model, best_score = self.optimized_model_selection(X_scaled, y)
        
        total_time = time.time() - start_time
        rmse = np.sqrt(best_score)
        
        print(f"\nOptimized Development Complete!")
        print(f"Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best Model: {best_model_name}")
        print(f"CV RMSE: {rmse:.8f}")
        print(f"CV MSE: {best_score:.8f}")
        
        # Estimate production time
        estimated_production = total_time / self.subset_ratio
        print(f"Estimated production time: {estimated_production/60:.1f} minutes")
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'rmse': rmse,
            'mse': best_score,
            'feature_cols': feature_cols,
            'time': total_time,
            'estimated_production': estimated_production
        }

def main_optimized():
    """Main function optimized for your system"""
    print("OPTIMIZED KAGGLE PIPELINE")
    print("System: 16 cores, 16GB RAM")
    print("Expected: 30-45 min development, 2-3 hours production")
    print("=" * 60)
    
    try:
        train_df = pd.read_parquet('train.parquet')
        test_df = pd.read_parquet('test.parquet')
        print(f"Data loaded: Train {train_df.shape}, Test {test_df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Memory check
    train_memory = train_df.memory_usage(deep=True).sum() / 1024**3
    print(f"Train data memory: {train_memory:.2f} GB")
    
    # Create optimized predictor
    predictor = OptimizedIVPredictor(use_subset=True, subset_ratio=0.15)
    
    # Run development
    results = predictor.rapid_development(train_df)
    
    if results:
        print(f"\n🚀 Ready for production training!")
        print(f"Recommendation: Proceed with {results['model_name']} model")
        
        # Memory cleanup
        del train_df, test_df
        gc.collect()
        
        return results
    else:
        print("❌ Development failed")
        return None

if __name__ == "__main__":
    main_optimized()
