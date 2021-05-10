import argparse
import pandas as pd
from model import my_custom_accuracy
from preprocessing import parse_data
from joblib import load
from config import model_name, result_path
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t", index_col='id', parse_dates=['release_date'])

# Parse Data
parsed_data, parsed_label, parsed_index, parsed_log_label = parse_data(data, train=False)
model = load(model_name)
results = model.predict(parsed_data)
# results = np.expm1(results)
#results[results<0] = 0.0
print(f"Loss on data {-my_custom_accuracy(parsed_log_label, results)}")

prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = parsed_index
results = np.expm1(results)
prediction_df['revenue'] = results


# export prediction results
prediction_df.to_csv(result_path, index=False, header=False)



