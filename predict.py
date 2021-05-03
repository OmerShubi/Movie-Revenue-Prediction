import argparse
import pandas as pd

from model import my_custom_accuracy
from preprocessing import parse_data
from joblib import load
from config import model_name, result_path

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t", index_col='id', parse_dates=['release_date'])

# Parse Data
# TODO competition adjustments - no label
parsed_data, parsed_label, parsed_index = parse_data(data, train=False)
model = load(model_name)
results = model.predict(parsed_data)

prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = parsed_index
prediction_df['revenue'] = results

print(f"Loss on data {-my_custom_accuracy(parsed_label, results)}")

# export prediction results
prediction_df.to_csv(result_path, index=False, header=False)



