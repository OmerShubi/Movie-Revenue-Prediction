from tpot import TPOTRegressor
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from config import parsed_train_path, parsed_test_path, checkpoint_folder
import numpy as np
import logging
from pprint import pformat

# TODO torch
# TODO score instructions - make sure that the revenue prediction is >=0
# TODO return score

# TODO save model and load - check the results

def my_custom_accuracy(y_true, y_pred):
    y_pred[y_pred<0] = 0.0
    return -mean_squared_log_error(y_true, y_pred)


def create_and_configer_logger(log_name='log_file.log', level=logging.DEBUG):
    """
    Sets up a logger that works across files.
    The logger prints to console, and to log_name log file.

    Example usage:
        In main function:
            logger = create_and_configer_logger(log_name='myLog.log')
        Then in all other files:
            logger = logging.getLogger(_name_)

        To add records to log:
            logger.debug(f"New Log Message. Value of x is {x}")

    Args:
        log_name: str, log file name

    Returns: logger
    """
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
    logger = create_and_configer_logger(log_name='log_file.log', level=logging.INFO)

    with open(parsed_train_path, 'rb') as f:
        parsed_train_data = np.load(f)
        parsed_train_label = np.load(f)
    with open(parsed_test_path, 'rb') as f:
        parsed_test_data = np.load(f)
        parsed_test_label = np.load(f)

    logger.info("Finished loading data")
    my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)
    tpot = TPOTRegressor(generations=1, population_size=1, max_eval_time_mins=1,
                         max_time_mins=2, verbosity=2,
                         n_jobs=1, scoring=my_custom_scorer, log_file="log.log",
                         random_state=0,
                         periodic_checkpoint_folder = checkpoint_folder)
    tpot.fit(parsed_train_data, parsed_train_label)
    logger.info("Finished fitting the model")
    logger.info(f"The best pipeline \n {tpot.fitted_pipeline_}")
    logger.info(f"Score on test data {tpot.score(parsed_test_data, parsed_test_label)}")
    logger.info(f"Trials \n {pformat(tpot.evaluated_individuals_)}")
    tpot.export('best_model.py')



