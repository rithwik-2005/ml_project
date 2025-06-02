import os
import sys
import pickle
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("ModelTrainer initialized")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training process")
            
            # Verify input data
            if train_array.size == 0 or test_array.size == 0:
                raise ValueError("Empty training or test array received")
            
            # Split data
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            # Define models with DEFAULT parameters (no lists)
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100),
                "Decision Tree": DecisionTreeRegressor(criterion='squared_error'),
                "Gradient Boosting": GradientBoostingRegressor(learning_rate=0.1, n_estimators=100),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(learning_rate=0.1, n_estimators=100),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, train_dir="catboost_logs"),
                "AdaBoost Regressor": AdaBoostRegressor(learning_rate=0.1, n_estimators=100),
                "K-Neighbors Regressor": KNeighborsRegressor(),
            }
            
            # Parameter grids for GridSearchCV (separate from model initialization)
            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 9]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 9]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }

            # Evaluate models with GridSearchCV
            from sklearn.model_selection import GridSearchCV
            
            model_report = {}
            best_models = {}
            
            for model_name, model in models.items():
                try:
                    logging.info(f"Training {model_name}")
                    
                    if model_name in params and params[model_name]:
                        # Perform grid search if parameters exist
                        grid_search = GridSearchCV(
                            estimator=model,
                            param_grid=params[model_name],
                            cv=3,
                            scoring='r2',
                            n_jobs=-1
                        )
                        grid_search.fit(X_train, y_train)
                        
                        # Get best model and score
                        best_model = grid_search.best_estimator_
                        best_score = grid_search.best_score_
                        
                        best_models[model_name] = best_model
                        model_report[model_name] = best_score
                        
                        logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
                        logging.info(f"{model_name} best R2 score: {best_score:.4f}")
                    else:
                        # Train without grid search
                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)
                        model_report[model_name] = score
                        best_models[model_name] = model
                        logging.info(f"{model_name} R2 score: {score:.4f}")
                        
                except Exception as e:
                    logging.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            if not model_report:
                raise ValueError("No model evaluation results returned")
            
            # Select best model
            best_model_name = max(model_report.items(), key=lambda x: x[1])[0]
            best_model_score = model_report[best_model_name]
            best_model = best_models[best_model_name]
            
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}")
            
            # Verify minimum performance threshold
            if best_model_score < 0.6:
                raise ValueError(f"No model met minimum R2 score (0.6). Best was {best_model_score:.4f}")
            
            # Save the best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)
            
            logging.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}")
            
            # Calculate final R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Training completed. Final R2 score: {r2_square:.4f}")
            
            return r2_square
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)