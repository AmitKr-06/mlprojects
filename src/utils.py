import sys
import os
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
def evaluate_models(X_train, y_train, X_test, y_test, models, param, problem_type='classification'):
    try:
        report = {}
        
        # Choose metric based on problem type
        if problem_type == 'classification':
            from sklearn.metrics import accuracy_score
            metric = accuracy_score
        else:
            from sklearn.metrics import r2_score
            metric = r2_score
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            # Only run GridSearch if parameters are provided
            if para:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_test_pred = model.predict(X_test)
            
            # Calculate score
            test_model_score = metric(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)