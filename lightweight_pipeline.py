import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

class LightweightIVPredictor:
    """Lightweight predictor using only scikit-learn (no LightGBM/XGBoost)"""
    
    def __init__(self, use_subset=True, subset_ratio=0.1):
        self.use_subset = use_subset
        self.subset_ratio = subset_ratio
        self.scaler = RobustScaler()
        self.models = {}
        
    def create_subset(self, df):
        """Create development subset"""
        if not self.use_subset:
            return df
        subset_size = max(1000, int(len(df) * self.subset_ratio))
        return df.sample(n=subset_size, random_state=42)
    
    def fast_feature_engineering(self, df):
        """Basic feature engineering without complex dependencies"""
        # Clean IV data
        iv_cols = [col for col in df.columns if '_iv_' in col]
        for col in iv_cols:
            if col in df.columns:
                df[col] = df[col].clip(0.001, 5.0)
        
        # Basic surface features
        if iv_cols:
            df['iv_mean'] = df[iv_cols].mean(axis=1, skipna=True)
            df['iv_std'] = df[iv_cols].std(axis=1, skipna=True)
            df['iv_range'] = df[iv_cols].max(axis=1) - df[iv_cols].min(axis=1)
        
        # X features
        x_cols = [col for col in df.columns if col.startswith('X')][:15]
        if x_cols:
            df['X_mean'] = df[x_cols].mean(axis=1)
            df['X_std'] = df[x_cols].std(axis=1)
        
        # Moneyness for key strikes
        if 'underlying' in df.columns:
            key_strikes = [24000, 25000, 26000]
            for strike in key_strikes:
                df[f'moneyness_{strike}'] = np.log(strike / df['underlying'])
        
        return df
    
    def train_lightweight_models(self, X, y):
        """Train models using only scikit-learn"""
        models = {
            'rf_fast': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'rf_deep': RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        best_score = float('inf')
        best_model = None
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            mse = -cv_scores.mean()
            print(f"{name}: MSE = {mse:.8f}")
            
            if mse < best_score:
                best_score = mse
                best_model = model
        
        return best_model, best_score
    
    def rapid_development(self, train_df):
        """Lightweight development cycle"""
        print("LIGHTWEIGHT DEVELOPMENT CYCLE")
        print("=" * 40)
        
        start_time = time.time()
        
        # Create subset
        train_subset = self.create_subset(train_df)
        print(f"Using {len(train_subset)} samples")
        
        # Feature engineering
        train_features = self.fast_feature_engineering(train_subset)
        
        # Target columns
        target_cols = [col for col in train_features.columns if '_iv_' in col]
        
        # Base features
        exclude_cols = ['timestamp', 'expiry'] + target_cols
        feature_cols = [col for col in train_features.columns if col not in exclude_cols]
        
        # Train on main target
        main_target = 'call_iv_25000' if 'call_iv_25000' in target_cols else target_cols[0]
        mask = train_features[main_target].notna()
        
        if mask.sum() < 50:
            print(f"Insufficient data")
            return None
        
        X = train_features.loc[mask, feature_cols].fillna(0)
        y = train_features.loc[mask, main_target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        best_model, best_score = self.train_lightweight_models(X_scaled, y)
        
        total_time = time.time() - start_time
        
        print(f"\nDevelopment Complete!")
        print(f"Time: {total_time:.1f}s")
        print(f"Best MSE: {best_score:.8f}")
        
        return {
            'model': best_model,
            'mse': best_score,
            'feature_cols': feature_cols,
            'time': total_time
        }

def main_lightweight():
    """Main function for lightweight approach"""
    print("LIGHTWEIGHT KAGGLE APPROACH")
    print("=" * 40)
    
    # Check if data exists
    try:
        train_df = pd.read_parquet('train.parquet')
        test_df = pd.read_parquet('test.parquet')
    except:
        print("❌ Data files not found!")
        print("Make sure train.parquet and test.parquet are in the current directory")
        return
    
    # Create predictor
    predictor = LightweightIVPredictor(use_subset=True, subset_ratio=0.1)
    
    # Run development
    results = predictor.rapid_development(train_df)
    
    if results:
        print(f"\n✅ Ready for submission!")
        print(f"Estimated full training time: {results['time'] / 0.1 / 60:.1f} minutes")
    
if __name__ == "__main__":
    main_lightweight()
