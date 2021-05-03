import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from config import parsed_train_path, parsed_test_path
from sklearn.metrics import mean_squared_log_error
from model import my_custom_accuracy
from joblib import dump, load


with open(parsed_train_path, 'rb') as f:
    parsed_train_data = np.load(f)
    parsed_train_label = np.load(f)
with open(parsed_test_path, 'rb') as f:
    parsed_test_data = np.load(f)
    parsed_test_label = np.load(f)

# Change the model
exported_pipeline = DecisionTreeRegressor(max_depth=2, min_samples_leaf=7, min_samples_split=9)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 0)

exported_pipeline.fit(parsed_train_data, parsed_train_label)
dump(exported_pipeline, 'best_model.joblib')
results = exported_pipeline.predict(parsed_test_data)
print(f"Loss on test data {-my_custom_accuracy(parsed_test_label, results)}")
