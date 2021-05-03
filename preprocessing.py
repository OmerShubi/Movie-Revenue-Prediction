from ast import literal_eval
import argparse
import numpy as np
import pandas as pd
import csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
import os
import pickle
from datetime import datetime
from sklearn.pipeline import Pipeline
import joblib
from sklearn.compose import ColumnTransformer, make_column_transformer


def treat_dict_column(data, old_col_name, new_col_name, key):
    data[old_col_name].fillna('{}', inplace = True)
    data[old_col_name] = data[old_col_name].apply(literal_eval)
    data[new_col_name] = data[old_col_name].apply(pd.Series)[key]
    data.drop(old_col_name, inplace=True, axis=1)
    return data

def treat_list_of_dicts_column_to_one_list(data, col_name, key):
    data[col_name] = data[col_name].apply(literal_eval)
    data[col_name] = data[col_name].apply(lambda x: [d[key] for d in x] if x != [] else ["nan"])
    return data

def treat_list_of_dicts_column_to_multiple_lists(data, col_name, keys):
    data[col_name] = data[col_name].apply(literal_eval)
    data[col_name] = data[col_name].apply(lambda x: [[d[key] for key in keys] for d in x])
    return data

def parse_data(data, max_order=2, train=True):
    data_label = data["revenue"]
    # remove features - unreasonable & 1 uniqe value features
    data.drop(["backdrop_path", "homepage", "imdb_id", "status", "poster_path", "revenue"], inplace=True, axis=1)

    # Flatten nested objects
    data = treat_dict_column(data, "belongs_to_collection", "collection_name", "name")
    data = treat_list_of_dicts_column_to_one_list(data, 'genres', 'name')
    data = treat_list_of_dicts_column_to_multiple_lists(data, 'cast', ['name', 'gender'])

    # TODO multi dummies
    data.drop(["production_companies", "production_countries", "spoken_languages", "crew"], inplace=True, axis=1)
    # data = treat_list_of_dicts_column_to_one_list(data, 'production_companies', 'name')
    # data = treat_list_of_dicts_column_to_one_list(data, 'production_countries', 'iso_3166_1')
    # data = treat_list_of_dicts_column_to_one_list(data, 'spoken_languages', 'iso_639_1')
    # data = treat_list_of_dicts_column_to_one_list(data, 'crew', 'name')
    multi_dummy_columns = ["genres"]  # ["production_companies", "production_countries", "spoken_languages", "crew"]

    # TODO embedding features
    #data = treat_list_of_dicts_column_to_one_list(data, 'Keywords', 'name')
    embedding_features = ["original_title", "overview", "title", "tagline", "Keywords"]
    data.drop(embedding_features, inplace=True, axis=1)

    #
    # data['cast_size'] = data['cast'].apply(lambda x: len(x))
    # data['cast_size'].hist(bins=50)
    # plt.savefig(f"cast_size.png", bbox_inches='tight')
    # data.drop("cast_size", inplace=True, axis=1)

    # Save only the first max_order cast info a split it into 2*max_order columns
    for i in range(max_order):
        data[f'cast_{i}_name'] = data['cast'].apply(lambda x: x[i][0] if len(x) > i else None)
        data[f'cast_{i}_gender'] = data['cast'].apply(lambda x: x[i][1] if len(x) > i else None)
    data.drop("cast", inplace=True, axis=1)

    # Convert Bool to 1 and 0
    data['video'] = data['video'].astype(int)

    # Convert release_date to timestamp
    data['release_date'] = data['release_date'].apply(lambda x: datetime.timestamp(x))

    numerical_columns = ["popularity", "budget", "runtime", "vote_average", "vote_count", "release_date"]
    dummy_columns = ["collection_name", "original_language"]
    for i in range(max_order):
        dummy_columns.append(f'cast_{i}_name')
        dummy_columns.append(f'cast_{i}_gender')

    if train:
        # Normalized numerical features
        pipe = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                         ('standard_scaler', StandardScaler()), ('minmax_scaler', MinMaxScaler(clip=True))])
        pipe.fit(data[numerical_columns].to_numpy())
        joblib.dump(pipe, 'pipeline_scaler.pkl')

        # One hot Encoders fit
        encoders = {}
        for col_d in dummy_columns:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(data[col_d].to_numpy().reshape(-1, 1))
            encoders[col_d] = enc
        for col_m in multi_dummy_columns:
            enc = MultiLabelBinarizer()
            enc.fit((data[col_m]))
            encoders[col_m] = enc
        joblib.dump(encoders, 'encoders.pkl')
    else:
        pipe = joblib.load('pipeline_scaler.pkl')
        encoders = joblib.load('encoders.pkl')

    # Normalized numerical features
    data[numerical_columns] = pipe.transform(data[numerical_columns].to_numpy())

    # One hot Encoders transform
    data_arr_dummies = []
    for col_d in dummy_columns:
        enc = encoders[col_d]
        data_arr_dummies.append(enc.transform(data[col_d].to_numpy().reshape(-1, 1)).toarray())
        data.drop(col_d, inplace=True, axis=1)
    for col_m in multi_dummy_columns:
        enc = encoders[col_m]
        data_arr_dummies.append(enc.transform((data[col_m])))
        data.drop(col_m, inplace=True, axis=1)
    data_arr = np.concatenate([data.to_numpy()]+data_arr_dummies, axis=1)

    return data_arr, data_label.to_numpy()

if __name__ == '__main__':
    train_path = 'train.tsv'
    train_data = pd.read_csv(train_path, sep="\t", index_col='id', parse_dates=['release_date'])
    parsed_train_path = "parsed_train.pkl"
    if os.path.exists(parsed_train_path):
        parsed_train_data, parsed_train_label = pd.read_pickle(parsed_train_path)
    else:
        parsed_train_data, parsed_train_label = parse_data(train_data, train=True)
        # TODO uncomment
        # pd.to_pickle((parsed_train_data,parsed_train_label), parsed_train_path)

    test_path = 'test.tsv'
    test_data = pd.read_csv(test_path, sep="\t", index_col='id', parse_dates=['release_date'])
    parsed_test_path = "parsed_test.pkl"
    if os.path.exists(parsed_test_path):
        parsed_test_data, parsed_test_label = pd.read_pickle(parsed_test_path)
    else:
        parsed_test_data, parsed_test_label = parse_data(test_data, train=False)
        # TODO uncomment
        # pd.to_pickle((parsed_test_data,parsed_test_label), parsed_test_path)

    # TODO competition adjustments - no label
    print(1)
