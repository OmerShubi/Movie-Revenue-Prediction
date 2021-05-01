from ast import literal_eval
import argparse
import numpy as np
import pandas as pd
import csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pickle
from datetime import datetime
from sklearn.pipeline import Pipeline


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
    # remove features - unreasonable & 1 uniqe value features
    data.drop(["backdrop_path", "homepage", "imdb_id", "status", "poster_path"], inplace=True, axis=1)

    # Flatten nested objects
    data = treat_dict_column(data, "belongs_to_collection", "collection_name", "name")
    data = treat_list_of_dicts_column_to_one_list(data, 'genres', 'name')
    data = treat_list_of_dicts_column_to_one_list(data, 'production_companies', 'name')
    data = treat_list_of_dicts_column_to_one_list(data, 'spoken_languages', 'iso_639_1')
    data = treat_list_of_dicts_column_to_one_list(data, 'Keywords', 'name')
    data = treat_list_of_dicts_column_to_one_list(data, 'crew', 'name')
    data = treat_list_of_dicts_column_to_one_list(data, 'production_countries', 'iso_3166_1')
    data = treat_list_of_dicts_column_to_multiple_lists(data, 'cast', ['name', 'gender'])

    #
    # data['cast_size'] = data['cast'].apply(lambda x: len(x))
    # data['cast_size'].hist(bins=50)
    # plt.savefig(f"cast_size.png", bbox_inches='tight')
    # data.drop("cast_size", inplace=True, axis=1)

    # Save only the first max_order cast info a split it into 2*max_order columns
    for i in range(max_order):
        data[f'cast_{i}_name'] = data['cast'].apply(lambda x: x[i][0] if len(x) > i else None)
        data[f'cast_{i}_gender'] = data['cast'].apply(lambda x: x[i][1] - 1 if len(x) > i else None)
    data.drop("cast", inplace=True, axis=1)

    # Create dummy variables
    multi_dummy_columns = ["genres", "production_companies", "production_countries", "spoken_languages", "crew"]
    for col in multi_dummy_columns:
        data = data.join(data[col].str.join(sep='*').str.get_dummies(sep='*').add_prefix(f"{col}_")).drop(col, inplace=False, axis=1)
    dummy_columns = ["collection_name", "original_language"]
    for i in range(max_order):
        dummy_columns.append(f'cast_{i}_name')
    data = pd.get_dummies(data, columns=dummy_columns, dummy_na=True)

    # Convert Bool to 1 and 0
    data['video'] = data['video'].astype(int)

    if train:
        # Normalized numerical features
        numerical_columns = ["popularity", "budget", "runtime", "vote_average", "vote_count"]
        standard_scaler = StandardScaler()
        data[numerical_columns] = standard_scaler.fit_transform(data[numerical_columns].to_numpy())
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
        minmax_scaler = MinMaxScaler()
        data[numerical_columns] = minmax_scaler.fit_transform(data[numerical_columns].to_numpy())
        # TODO save StandardScaler & MinMaxScaler
        with open('standard_scaler.pickle', 'wb') as file:
            pickle.dump(standard_scaler, file)
        with open('minmax_scaler.pickle', 'wb') as file:
            pickle.dump(minmax_scaler, file)
    else:
        with open('standard_scaler.pickle', 'rb') as file:
            standard_scaler = pickle.load(file)
        with open('minmax_scaler.pickle', 'rb') as file:
            minmax_scaler = pickle.load(file)
        # TODO use pretrained scaler
        pass

    # Convert release_date to timestamp
    data['release_date'] = data['release_date'].apply(lambda x: datetime.timestamp(x))

    return data

if __name__ == '__main__':
    train_path = 'train.tsv'
    train_data = pd.read_csv(train_path, sep="\t", index_col='id', parse_dates=['release_date'])
    parsed_train_path = "parsed_train_data.pkl"
    if os.path.exists(parsed_train_path):
        parsed_train_data = pd.read_pickle(parsed_train_path)
    else:
        parsed_train_data = parse_data(train_data)
    print(1)