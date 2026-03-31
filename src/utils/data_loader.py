import os
import sys
import pandas as pd
import yaml
from src.exception import CustomException
from src.logger import logging

class DataLoader:
    """Flexible data loader for multiple sources"""
    
    def __init__(self, config_path='config/data_config.yaml'):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logging.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logging.warning(f"Could not load config file: {e}")
            # Default configuration
            self.config = {
                'data_source': {
                    'type': 'csv',
                    'path': 'notebook/data/titanic_train.csv'
                },
                'split_config': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'shuffle': True
                }
            }
    
    def load_data(self):
        """Load data based on configuration"""
        try:
            source_type = self.config['data_source']['type']
            
            if source_type == 'csv':
                df = self._load_from_csv()
            elif source_type == 'mongodb':
                df = self._load_from_mongodb()
            elif source_type == 'sql':
                df = self._load_from_sql()
            elif source_type == 'api':
                df = self._load_from_api()
            else:
                raise ValueError(f"Unsupported data source type: {source_type}")
            
            logging.info(f"Data loaded successfully from {source_type} source")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_from_csv(self):
        """Load data from CSV file"""
        file_path = self.config['data_source']['path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def _load_from_mongodb(self):
        """Load data from MongoDB"""
        try:
            from pymongo import MongoClient
            
            mongo_config = self.config['data_source']['mongodb']
            client = MongoClient(mongo_config['uri'])
            db = client[mongo_config['database']]
            collection = db[mongo_config['collection']]
            
            # Load data into pandas
            cursor = collection.find({})
            df = pd.DataFrame(list(cursor))
            
            # Remove MongoDB _id column if exists
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            client.close()
            return df
            
        except ImportError:
            raise ImportError("pymongo not installed. Run: pip install pymongo")
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_from_sql(self):
        """Load data from SQL database"""
        try:
            from sqlalchemy import create_engine
            
            sql_config = self.config['data_source']['sql']
            engine = create_engine(sql_config['connection_string'])
            df = pd.read_sql(sql_config['table'], engine)
            engine.dispose()
            return df
            
        except ImportError:
            raise ImportError("sqlalchemy not installed. Run: pip install sqlalchemy")
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_from_api(self):
        """Load data from REST API"""
        try:
            import requests
            
            api_config = self.config['data_source']['api']
            response = requests.get(api_config['url'])
            
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data)
            else:
                raise Exception(f"API returned status code: {response.status_code}")
                
        except ImportError:
            raise ImportError("requests not installed. Run: pip install requests")
        except Exception as e:
            raise CustomException(e, sys)