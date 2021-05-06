import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from config import parsed_train_path, parsed_test_path, model_name
from sklearn.metrics import mean_squared_log_error
from model import my_custom_accuracy
from joblib import dump, load
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive



### Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


if __name__ == '__main__':

    with open(parsed_train_path, 'rb') as f:
        parsed_train_data = np.load(f)
        parsed_train_label = np.load(f)
    with open(parsed_test_path, 'rb') as f:
        parsed_test_data = np.load(f)
        parsed_test_label = np.load(f)

    # Change the model
    # Average CV score on the training set was: -2.6181800521534355
    exported_pipeline = make_pipeline(
        SelectFwe(score_func=f_regression, alpha=0.048),
        StackingEstimator(estimator=LassoLarsCV(normalize=False)),
        DecisionTreeRegressor(max_depth=10, min_samples_leaf=7, min_samples_split=20)
    )

    import xgboost as xgb
    # exported_pipeline = make_pipeline(xgb.XGBRegressor(max_depth=5,
    #                             learning_rate=0.01,
    #                             n_estimators=100,
    #                             objective='reg:squarederror',
    #                             gamma=1.45,
    #                             seed=1,
    #                             verbosity=2,
    #                             subsample=0.8,
    #                             colsample_bytree=0.7,
    #                             colsample_bylevel=0.5))


    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 1)

    exported_pipeline.fit(parsed_train_data, parsed_train_label)
    dump(exported_pipeline, model_name)


    results_train = exported_pipeline.predict(parsed_train_data)
    print(f"Loss on train data {-my_custom_accuracy(parsed_train_label, results_train)}")

    results_test = exported_pipeline.predict(parsed_test_data)
    print(f"Loss on test data {-my_custom_accuracy(parsed_test_label, results_test)}")

    # ### Example - Calculating RMSLE
    # res = rmsle(parsed_test_label, results)
    # print("RMSLE is: {:.6f}".format(res))
