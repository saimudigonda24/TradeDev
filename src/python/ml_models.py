import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from dataclasses import dataclass
import joblib

@dataclass
class ModelResult:
    predictions: np.ndarray
    confidence_intervals: Tuple[np.ndarray, np.ndarray]
    feature_importance: Dict[str, float]
    metrics: Dict[str, float]
    model_name: str

class MLModels:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        
    def prepare_data(self, data: pd.DataFrame, 
                    target_col: str = 'Close',
                    forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training"""
        # Create target variable (future prices)
        data = data.copy()
        data[f'{target_col}_future'] = data[target_col].shift(-forecast_horizon)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col not in [f'{target_col}_future', target_col]]
        X = data[feature_cols].values
        y = data[f'{target_col}_future'].values
        
        return X, y, feature_cols
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str]) -> RandomForestRegressor:
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        self.trained_models['random_forest'] = {
            'model': model,
            'feature_names': feature_names
        }
        
        return model
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray, 
                     feature_names: List[str]) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        self.trained_models['xgboost'] = {
            'model': model,
            'feature_names': feature_names
        }
        
        return model
    
    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str]) -> GradientBoostingRegressor:
        """Train Gradient Boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X, y)
        self.trained_models['gradient_boosting'] = {
            'model': model,
            'feature_names': feature_names
        }
        
        return model
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      feature_names: List[str]) -> Dict:
        """Train ensemble of models"""
        models = {
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=50, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X, y)
            trained_models[name] = model
        
        self.trained_models['ensemble'] = {
            'models': trained_models,
            'feature_names': feature_names
        }
        
        return trained_models
    
    def predict_with_confidence(self, model_name: str, X: np.ndarray, 
                               confidence_level: float = 0.95) -> ModelResult:
        """Make predictions with confidence intervals"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        model_info = self.trained_models[model_name]
        
        if model_name == 'ensemble':
            predictions = self._ensemble_predict(model_info['models'], X)
            feature_importance = self._ensemble_feature_importance(model_info['models'], 
                                                                 model_info['feature_names'])
        else:
            model = model_info['model']
            predictions = model.predict(X)
            feature_importance = dict(zip(model_info['feature_names'], 
                                        model.feature_importances_))
        
        # Calculate confidence intervals using quantile regression or bootstrap
        lower_bound, upper_bound = self._calculate_confidence_intervals(
            model_name, X, predictions, confidence_level
        )
        
        return ModelResult(
            predictions=predictions,
            confidence_intervals=(lower_bound, upper_bound),
            feature_importance=feature_importance,
            metrics={},  # Will be filled during evaluation
            model_name=model_name
        )
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        result = self.predict_with_confidence(model_name, X_test)
        predictions = result.predictions
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      feature_names: List[str], n_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        models = {
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=50, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            fold_scores = {'mse': [], 'mae': [], 'r2': []}
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                fold_scores['mse'].append(mean_squared_error(y_test, predictions))
                fold_scores['mae'].append(mean_absolute_error(y_test, predictions))
                fold_scores['r2'].append(r2_score(y_test, predictions))
            
            results[model_name] = {
                'mse_mean': np.mean(fold_scores['mse']),
                'mse_std': np.std(fold_scores['mse']),
                'mae_mean': np.mean(fold_scores['mae']),
                'mae_std': np.std(fold_scores['mae']),
                'r2_mean': np.mean(fold_scores['r2']),
                'r2_std': np.std(fold_scores['r2'])
            }
        
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                             model_type: str = 'xgboost') -> Dict:
        """Perform hyperparameter tuning"""
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def _ensemble_predict(self, models: Dict, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        for model in models.values():
            predictions.append(model.predict(X))
        
        # Simple average ensemble
        return np.mean(predictions, axis=0)
    
    def _ensemble_feature_importance(self, models: Dict, 
                                   feature_names: List[str]) -> Dict[str, float]:
        """Calculate ensemble feature importance"""
        importance_sum = np.zeros(len(feature_names))
        
        for model in models.values():
            if hasattr(model, 'feature_importances_'):
                importance_sum += model.feature_importances_
        
        importance_avg = importance_sum / len(models)
        return dict(zip(feature_names, importance_avg))
    
    def _calculate_confidence_intervals(self, model_name: str, X: np.ndarray, 
                                      predictions: np.ndarray, 
                                      confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        # Simplified approach using prediction variance
        # In practice, you might use quantile regression or bootstrap methods
        
        if model_name in self.trained_models:
            model_info = self.trained_models[model_name]
            
            if model_name == 'ensemble':
                # For ensemble, calculate prediction variance across models
                all_predictions = []
                for model in model_info['models'].values():
                    all_predictions.append(model.predict(X))
                
                prediction_std = np.std(all_predictions, axis=0)
            else:
                # For single models, use a simple heuristic
                prediction_std = np.std(predictions) * np.ones_like(predictions)
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin_of_error = z_score * prediction_std
        
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        return lower_bound, upper_bound
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], filepath)
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        self.trained_models[model_name] = joblib.load(filepath)
