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
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import xgboost as xgb


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

    #exported_pipeline = define_model()
    # Change the model
    # # Average CV score on the training set was: -2.6181800521534355
    # exported_pipeline = make_pipeline(
    #     SelectFwe(score_func=f_regression, alpha=0.048),
    #     StackingEstimator(estimator=LassoLarsCV(normalize=False)),
    #     DecisionTreeRegressor(max_depth=10, min_samples_leaf=7, min_samples_split=20)
    # )
    # exported_pipeline = make_pipeline(
    #     SelectFwe(score_func=f_regression, alpha=0.001),
    #     StackingEstimator(
    #         estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.35000000000000003, min_samples_leaf=17,
    #                                       min_samples_split=18, n_estimators=50)),
    #     SelectFwe(score_func=f_regression, alpha=0.044),
    #     SelectFwe(score_func=f_regression, alpha=0.017),
    #     StackingEstimator(
    #         estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.35000000000000003, min_samples_leaf=17,
    #                                       min_samples_split=18, n_estimators=100)),
    #     XGBRegressor(learning_rate=0.01, max_depth=6, min_child_weight=10, n_estimators=50, n_jobs=1,
    #                  objective="reg:squarederror", subsample=1.0, verbosity=0)
    # )

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

    # Average CV score on the training set was: -2.315083207257054



    # exported_pipeline = make_pipeline(
    #     StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.55, min_samples_leaf=17,
    #                                                       min_samples_split=15, n_estimators=50, random_state=1)),
    #     StackingEstimator(
    #         estimator=RandomForestRegressor(bootstrap=True, max_features=1.0, min_samples_leaf=1,
    #                                         min_samples_split=14,n_estimators=25,random_state=1)),
    #     RandomForestRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=6,
    #                           min_samples_split=14, n_estimators=25, random_state=1)
    # )

    # Average CV score on the training set was: -2.315303012402409
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.55, min_samples_leaf=17,
                                                          min_samples_split=15, n_estimators=50, random_state=1)),
        StackingEstimator(
            estimator=RandomForestRegressor(bootstrap=True, max_features=1.0, min_samples_leaf=1,
                                            min_samples_split=8,
                                            n_estimators=25, random_state=1)),
        RandomForestRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=6,
                              min_samples_split=14, n_estimators=25, random_state=1)
    )

    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 1)


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
