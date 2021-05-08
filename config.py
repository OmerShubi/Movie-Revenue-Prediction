import numpy as np

parsed_train_path = "data/parsed_train.npy"
parsed_test_path = "data/parsed_test.npy"
checkpoint_folder = "check-points"
model_name = 'data/best_model.joblib'
result_path = "prediction.csv"
scaler_path = 'data/pipeline_scaler.pkl'
encoder_path = 'data/encoders.pkl'
train_path = 'train.tsv'
test_path = 'test.tsv'
max_values = 1500
our_log_path = 'log/training_log.log'
tpot_log_path = "log/tpot_log.log"
stop_words = {'ourselves', 'hers', 'between',
              'yourself', 'but', 'again', 'there',
              'about', 'once', 'during', 'out', 'very',
              'having', 'with', 'they', 'own', 'an',
              'be', 'some', 'for', 'do', 'its', 'yours',
              'such', 'into', 'of', 'most', 'itself', 'other',
              'off', 'is', 's', 'am', 'or', 'who', 'as', 'from',
              'him', 'each', 'the', 'themselves', 'until', 'below',
              'are', 'we', 'these', 'your', 'his', 'through', 'don',
              'nor', 'me', 'were', 'her', 'more', 'himself', 'this',
              'down', 'should', 'our', 'their', 'while', 'above',
              'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no',
              'when', 'at', 'any', 'before', 'them', 'same', 'and',
              'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
              'then', 'that', 'because', 'what', 'over', 'why', 'so',
              'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself',
              'has', 'just', 'where', 'too', 'only', 'myself', 'which',
              'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if',
              'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
              'further', 'was', 'here', 'than'}

generations = None
population_size = 10
max_eval_time_mins = 20
max_time_mins = 2160
n_jobs = 3
custom_regressor_config_dict = {
    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [25, 50, 100, 200],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [25, 50, 100, 200],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [25, 50, 100, 200],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [25, 50, 100, 200],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [25, 50, 100, 200],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0],
        'objective': ['reg:squarederror', 'reg:squaredlogerror']
    },

    'sklearn.linear_model.SGDRegressor': {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    # Preprocessors
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'tpot.builtins.ZeroCount': {
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.1, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [25, 50, 100, 200],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }
}
