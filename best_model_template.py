from config import parsed_train_path, parsed_test_path, model_name
from model import my_custom_accuracy
from joblib import dump
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
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
        parsed_train_log_label = np.load(f)
    with open(parsed_test_path, 'rb') as f:
        parsed_test_data = np.load(f)
        parsed_test_label = np.load(f)
        parsed_test_log_label = np.load(f)

    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.01, loss="ls", max_depth=10,
                                                              max_features=0.9000000000000001, min_samples_leaf=20,
                                                              min_samples_split=8, n_estimators=200, subsample=0.4,
                                                              verbose=1)),
        RandomForestRegressor(bootstrap=True, max_features=0.7000000000000001, min_samples_leaf=4, min_samples_split=20,
                              n_estimators=200, verbose=1)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 1)

    #exported_pipeline.fit(parsed_train_data, parsed_train_label)
    exported_pipeline.fit(parsed_train_data, parsed_train_log_label)
    dump(exported_pipeline, model_name)

    #results_train = exported_pipeline.predict(parsed_train_data)
    #print(f"Loss on train data - regular revenue {-my_custom_accuracy(parsed_train_label, results_train)}")
    results_train_log = exported_pipeline.predict(parsed_train_data)
    #results_train = np.expm1(results_train_log)
    print(f"Loss on train data - log revenue {-my_custom_accuracy(parsed_train_log_label, results_train_log)}")

    #results_test = exported_pipeline.predict(parsed_test_data)
    #print(f"Loss on test data - regular revenue {-my_custom_accuracy(parsed_test_label, results_test)}")
    results_test_log = exported_pipeline.predict(parsed_test_data)
    #results_test = np.expm1(results_test_log)
    print(f"Loss on test data - log revenue {-my_custom_accuracy(parsed_test_log_label, results_test_log)}")

    # ### Example - Calculating RMSLE
    # res = rmsle(parsed_test_label, results)
    # print("RMSLE is: {:.6f}".format(res))
