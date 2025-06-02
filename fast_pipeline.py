import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error
import optuna
import time
import gc
import warnings
warnings.filterwarnings('ignore')

class FastIVPredictor:
    """Ultra-fast IV prediction system for rapid experimentation"""
    
    def __init__(self, use_subset=True, subset_ratio=0.1, max_features=50):
        self.use_subset = use_subset
        self.subset_ratio = subset_ratio
        self.max_features = max_features
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.best_model = None
        self.experiment_log = []
        
    def create_subset(self, df):
        """Create development subset for fast iteration"""
        if not self.use_subset:
            return df
            
        subset_size = max(1000, int(len(df) * self.subset_ratio))
        subset = df.sample(n=subset_size, random_state=42)
        print(f"Using subset: {len(subset)} rows ({len(subset)/len(df)*100:.1f}%)")
        return subset
    
    def fast_feature_engineering(self, df):
        """Lightning-fast feature engineering for development"""
        start_time = time.time()
        
        # Clean data quickly
        iv_cols = [col for col in df.columns if '_iv_' in col]
        for col in iv_cols:
            if col in df.columns:
                df[col] = df[col].clip(0.001, 5.0)
        
        # Basic volatility surface features (fast computation)
        if iv_cols:
            df['iv_mean'] = df[iv_cols].mean(axis=1, skipna=True)
            df['iv_std'] = df[iv_cols].std(axis=1, skipna=True)
            df['iv_median'] = df[iv_cols].median(axis=1, skipna=True)
            df['iv_range'] = df[iv_cols].max(axis=1) - df[iv_cols].min(axis=1)
        
        # X feature engineering (limited for speed)
        x_cols = [col for col in df.columns if col.startswith('X')][:20]  # Only top 20
        if x_cols:
            df['X_mean'] = df[x_cols].mean(axis=1)
            df['X_std'] = df[x_cols].std(axis=1)
        
        # Moneyness features for major strikes only
        major_strikes = [24000, 24500, 25000, 25500, 26000]
        if 'underlying' in df.columns:
            for strike in major_strikes:
                df[f'moneyness_{strike}'] = np.log(strike / df['underlying'])
        
        print(f"Feature engineering: {time.time() - start_time:.1f}s")
        return df
    
    def fast_model_selection(self, X, y):
        """Quick model selection with minimal hyperparameter tuning"""
        start_time = time.time()
        
        models = {
            'lgb_fast': lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                num_leaves=31, random_state=42, verbose=-1
            ),
            'xgb_fast': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            'rf_fast': RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
            )
        }
        
        best_score = float('inf')
        best_model_name = None
        
        for name, model in models.items():
            # Quick 3-fold CV
            cv_scores = cross_val_score(
                model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            mse = -cv_scores.mean()
            
            print(f"{name}: MSE = {mse:.8f}")
            
            if mse < best_score:
                best_score = mse
                best_model_name = name
        
        print(f"Model selection: {time.time() - start_time:.1f}s")
        print(f"Best model: {best_model_name} (MSE: {best_score:.8f})")
        
        return best_model_name, models[best_model_name]
    
    def optimize_hyperparameters(self, X, y, model_type):
        """Fast hyperparameter optimization"""
        start_time = time.time()
        
        def objective(trial):
            if model_type == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = xgb.XGBRegressor(**params)
            
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        print(f"Hyperparameter optimization: {time.time() - start_time:.1f}s")
        return study.best_params
    
    def prepare_data(self, df, target_cols):
        """Prepare data for training"""
        # Feature engineering
        df_features = self.fast_feature_engineering(df)
        
        # Select features
        exclude_cols = ['timestamp', 'expiry'] + target_cols
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Feature selection for speed
        if len(feature_cols) > self.max_features:
            print(f"Selecting top {self.max_features} features...")
            
            # Use first target for feature selection
            mask = df_features[target_cols[0]].notna()
            if mask.sum() > 100:
                X_temp = df_features.loc[mask, feature_cols].fillna(0)
                y_temp = df_features.loc[mask, target_cols[0]]
                
                self.feature_selector = SelectKBest(
                    mutual_info_regression, k=self.max_features
                )
                self.feature_selector.fit(X_temp, y_temp)
                
                selected_features = [feature_cols[i] for i in self.feature_selector.get_support(indices=True)]
                feature_cols = selected_features
        
        # Prepare final data
        X = df_features[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_cols
    
    def rapid_development(self, train_df):
        """Complete rapid development cycle"""
        print("Starting Rapid Development Cycle")
        print("=" * 50)
        
        start_time = time.time()
        
        # Create subset
        if self.use_subset:
            train_subset = self.create_subset(train_df)
        else:
            train_subset = train_df
        
        # Get target columns
        target_cols = [col for col in train_subset.columns if '_iv_' in col]
        print(f"Found {len(target_cols)} IV targets")
        
        # Prepare data
        X, feature_cols = self.prepare_data(train_subset, target_cols)
        
        # Train on main target (use call_iv_25000 as representative)
        main_target = 'call_iv_25000' if 'call_iv_25000' in target_cols else target_cols[0]
        mask = train_subset[main_target].notna()
        
        if mask.sum() < 50:
            print(f"Insufficient data for {main_target}")
            return None
        
        X_train = X[mask]
        y_train = train_subset.loc[mask, main_target]
        
        print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features")
        
        # Model selection
        best_model_name, best_model = self.fast_model_selection(X_train, y_train)
        
        # Hyperparameter optimization
        if best_model_name.startswith('lgb'):
            best_params = self.optimize_hyperparameters(X_train, y_train, 'lgb')
            self.best_model = lgb.LGBMRegressor(**best_params)
        elif best_model_name.startswith('xgb'):
            best_params = self.optimize_hyperparameters(X_train, y_train, 'xgb')
            self.best_model = xgb.XGBRegressor(**best_params)
        else:
            self.best_model = best_model
        
        # Final training
        self.best_model.fit(X_train, y_train)
        
        # Validation
        cv_scores = cross_val_score(
            self.best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
        )
        final_mse = -cv_scores.mean()
        final_rmse = np.sqrt(final_mse)
        
        total_time = time.time() - start_time
        
        print(f"\nDevelopment Cycle Complete!")
        print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"CV RMSE: {final_rmse:.8f}")
        print(f"CV MSE: {final_mse:.8f}")
        
        # Log experiment
        self.experiment_log.append({
            'timestamp': time.time(),
            'model': best_model_name,
            'rmse': final_rmse,
            'mse': final_mse,
            'time_seconds': total_time,
            'features': len(feature_cols),
            'samples': len(X_train)
        })
        
        return {
            'model': self.best_model,
            'rmse': final_rmse,
            'mse': final_mse,
            'feature_cols': feature_cols,
            'time': total_time
        }
    
    def scale_to_production(self, train_df, dev_results):
        """Scale best model to full production data"""
        print("\nScaling to Production")
        print("=" * 30)
        
        start_time = time.time()
        
        # Use full dataset
        self.use_subset = False
        
        # Get all targets
        target_cols = [col for col in train_df.columns if '_iv_' in col]
        
        # Prepare full data
        X_full, _ = self.prepare_data(train_df, target_cols)
        
        production_models = {}
        
        # Train model for each target
        for i, target_col in enumerate(target_cols):
            if target_col not in train_df.columns:
                continue
                
            mask = train_df[target_col].notna()
            if mask.sum() < 100:
                continue
            
            print(f"[{i+1}/{len(target_cols)}] Training {target_col}...")
            
            X_target = X_full[mask]
            y_target = train_df.loc[mask, target_col]
            
            # Create production model with same config as dev
            if isinstance(dev_results['model'], lgb.LGBMRegressor):
                model = lgb.LGBMRegressor(
                    **dev_results['model'].get_params(),
                    n_estimators=1000,  # More estimators for production
                    verbose=-1
                )
            else:
                model = type(dev_results['model'])(**dev_results['model'].get_params())
            
            model.fit(X_target, y_target)
            production_models[target_col] = model
        
        total_time = time.time() - start_time
        print(f"\nProduction Training Complete: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        return production_models
