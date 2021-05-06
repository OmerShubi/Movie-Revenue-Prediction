from tpot import TPOTRegressor
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from config import parsed_train_path, parsed_test_path, checkpoint_folder, our_log_path, tpot_log_path
import numpy as np
import logging
from pprint import pformat


def my_custom_accuracy(y_true, y_pred):
    y_pred[y_pred<0] = 0.0
    return -np.sqrt(mean_squared_log_error(y_true, y_pred))


def create_and_configer_logger(log_name, level=logging.DEBUG):
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=level,
        format='\n' + '[%(asctime)s - %(levelname)s] {%(pathname)s:%(lineno)d} -' + '\n' + ' %(message)s' + '\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger()
    return logger


if __name__ == '__main__':
    logger = create_and_configer_logger(log_name=our_log_path, level=logging.INFO)

    with open(parsed_train_path, 'rb') as f:
        parsed_train_data = np.load(f)
        parsed_train_label = np.load(f)
    with open(parsed_test_path, 'rb') as f:
        parsed_test_data = np.load(f)
        parsed_test_label = np.load(f)

    logger.info("Finished loading data")
    generations = 50
    population_size = 10
    max_eval_time_mins = 5
    max_time_mins = None
    n_jobs = 1
    config_dict = "TPOT light"
    logger.info(f"Run params: {generations=}, {population_size=}, {max_eval_time_mins=}, {max_time_mins=}"
                f"{n_jobs=}, {config_dict=}")

    my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)
    tpot = TPOTRegressor(generations=generations,
                         population_size=population_size,
                         max_eval_time_mins=max_eval_time_mins,
                         max_time_mins=max_time_mins,
                         verbosity=3,
                         n_jobs=n_jobs,
                         scoring=my_custom_scorer,
                         random_state=1,
                         periodic_checkpoint_folder=checkpoint_folder,
                         config_dict=config_dict)

    tpot.fit(parsed_train_data, parsed_train_label)
    logger.info("Finished fitting the model")
    logger.info(f"The best pipeline \n {tpot.fitted_pipeline_}")
    logger.info(f"Loss on test data {-tpot.score(parsed_test_data, parsed_test_label)}")
    logger.info(f"Trials \n {pformat(tpot.evaluated_individuals_)}")
    tpot.export('best_model.py')



