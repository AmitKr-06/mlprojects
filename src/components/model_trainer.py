import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self, problem_type='classification'):
        """
        problem_type: 'classification' or 'regression'
        """
        self.model_trainer_config = ModelTrainerConfig()
        self.problem_type = problem_type
    
    def _get_models(self):
        """Return models based on problem type"""
        if self.problem_type == 'classification':
            return {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
            }
        else:  # regression
            return {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
            }
    
    def _get_params(self):
        """Return hyperparameters based on problem type"""
        if self.problem_type == 'classification':
            return {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.6, 0.8, 1.0],
                    'max_depth': [3, 5, 7]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                },
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "CatBoost Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.5],
                    'n_estimators': [50, 100, 200]
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                }
            }
        else:  # regression
            return {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.6, 0.8, 1.0],
                    'max_depth': [3, 5, 7]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.5],
                    'n_estimators': [50, 100, 200]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                }
            }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate metrics based on problem type"""
        if self.problem_type == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        else:  # regression
            return {
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Split training and test input data for {self.problem_type}")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = self._get_models()
            params = self._get_params()
            
            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params,
                problem_type=self.problem_type  # Pass to evaluate_models
            )
            
            # Get best model score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            # Check if model performance is acceptable
            threshold = 0.6 if self.problem_type == 'classification' else 0.5
            if best_model_score < threshold:
                raise CustomException(f"No good model found (best score: {best_model_score:.4f})")
            
            logging.info(f"Best found model: {best_model_name} with score: {best_model_score:.4f}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Make predictions and calculate metrics
            predicted = best_model.predict(X_test)
            metrics = self._calculate_metrics(y_test, predicted)
            
            logging.info(f"Test Results: {metrics}")
            
            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # For Classification (Titanic)
    trainer_classification = ModelTrainer(problem_type='classification')
    # result = trainer_classification.initiate_model_trainer(train_array, test_array)
    
    # For Regression (House prices)
    trainer_regression = ModelTrainer(problem_type='regression')
    # result = trainer_regression.initiate_model_trainer(train_array, test_array)