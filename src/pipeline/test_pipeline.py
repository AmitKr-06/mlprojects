import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class TestPipeline:
    def __init__(self, model_path='artifacts/model.pkl', preprocessor_path='artifacts/preprocessor.pkl'):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
    def load_model_and_preprocessor(self):
        """
        Load the trained model and preprocessor
        """
        try:
            logging.info("Loading model and preprocessor...")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")
            return model, preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_single(self, data):
        """
        Make prediction for a single sample
        
        Args:
            data: dict with features
        """
        try:
            model, preprocessor = self.load_model_and_preprocessor()
            
            # Convert dict to DataFrame
            df = pd.DataFrame([data])
            
            # Transform features
            features = preprocessor.transform(df)
            
            # Predict
            prediction = model.predict(features)
            probability = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            
            return {
                'prediction': int(prediction[0]),
                'probability': probability[0].tolist() if probability is not None else None
            }
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_batch(self, data):
        """
        Make predictions for multiple samples
        
        Args:
            data: DataFrame with features
        """
        try:
            model, preprocessor = self.load_model_and_preprocessor()
            
            # Transform features
            features = preprocessor.transform(data)
            
            # Predict
            predictions = model.predict(features)
            probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None
            }
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_on_test_data(self, test_path='artifacts/test.csv'):
        """
        Evaluate model on test data
        """
        try:
            logging.info("Loading test data...")
            test_df = pd.read_csv(test_path)
            
            # Separate features and target
            X_test = test_df.drop('Survived', axis=1)
            y_test = test_df['Survived']
            
            # Make predictions
            model, preprocessor = self.load_model_and_preprocessor()
            X_test_transformed = preprocessor.transform(X_test)
            predictions = model.predict(X_test_transformed)
            
            # Calculate accuracy
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(y_test, predictions)
            
            logging.info(f"Test Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))
            
            return accuracy
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the pipeline
    pipeline = TestPipeline()
    
    # Example single prediction (replace with actual features)
    sample_data = {
        'Pclass': 3,
        'Sex': 'male',
        'Age': 22,
        'SibSp': 1,
        'Parch': 0,
        'Fare': 7.25,
        'Embarked': 'S'
    }
    
    # Make prediction
    result = pipeline.predict_single(sample_data)
    print(f"\nSample Prediction: {result}")
    
    # Evaluate on test data (if available)
    if os.path.exists('artifacts/test.csv'):
        accuracy = pipeline.evaluate_on_test_data()
        print(f"\nTest Accuracy: {accuracy:.4f}")