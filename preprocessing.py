from ast import literal_eval
import argparse
import numpy as np
import pandas as pd
import csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
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

    # Convert Bool to 1 and 0
    data['video'] = data['video'].astype(int)

    # Convert release_date to timestamp
    data['release_date'] = data['release_date'].apply(lambda x: datetime.timestamp(x))

    numerical_columns = ["popularity", "budget", "runtime", "vote_average", "vote_count"]
    dummy_columns = ["collection_name", "original_language"]
    for i in range(max_order):
        dummy_columns.append(f'cast_{i}_name')

    if train:
        # Normalized numerical features
        pipe = Pipeline([('standard_scaler', StandardScaler()), ('minmax_scaler', MinMaxScaler())])
        pipe.fit(data[numerical_columns].to_numpy())
        joblib.dump(pipe, 'pipeline_scaler.pkl')

        # # TODO - change to on hot of sklearn & save it in train and use in test
        # # Create dummy variables
        # multi_dummy_columns = ["genres", "production_companies", "production_countries", "spoken_languages", "crew"]
        # for col in multi_dummy_columns:
        #     data = data.join(data[col].str.join(sep='*').str.get_dummies(sep='*').add_prefix(f"{col}_")).drop(col,
        #                                                                                                       inplace=False,
        #                                                                                                       axis=1)
        encoders = {}
        for col in dummy_columns:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(data[col].to_numpy().reshape(-1, 1))
            encoders[col] = enc
        joblib.dump(encoders, 'encoders.pkl')
        # data = pd.get_dummies(data, columns=dummy_columns, dummy_na=True)
        # preprocess = make_column_transformer((["collection_name", "original_language"], OneHotEncoder()))
        # preprocess.fit_transform(data[["collection_name", "original_language"]]).toarray()

        # encoders = OneHotEncoder(handle_unknown='ignore')
        # encoders.fit(data['collection_name'].unique().reshape(1, -1))
        # transformed = encoders.transform(data["collection_name"].to_numpy().reshape(-1, 1))
        # # Create a Pandas DataFrame of the hot encoded column
        # ohe_df = pd.DataFrame(transformed, columns=encoders.get_feature_names())
        # # concat with original data
        # data = pd.concat([data, ohe_df], axis=1).drop(["collection_name"], axis=1)

        # dummy_columns = ["collection_name", "original_language"]
        # for i in range(max_order):
        #     dummy_columns.append(f'cast_{i}_name')
        # #for col in dummy_columns:
        # encoders.fit(data["collection_name"])
        # for col in dummy_columns:
        #     transformed = encoders.transform(data["collection_name"].to_numpy().reshape(-1, 1))
        #     # Create a Pandas DataFrame of the hot encoded column
        #     ohe_df = pd.DataFrame(transformed, columns=encoders.get_feature_names())
        #     # concat with original data
        #     data = pd.concat([data, ohe_df], axis=1).drop(["collection_name"], axis=1)
        #     #pd.get_dummies(data, columns=dummy_columns, dummy_na=True)
        #     #encoders.fit(X)
        #     # TODO dummy_na=True
    else:
        # TODO use pretrained dummy variables
        # TODO check if dummy is not in the train
        pipe = joblib.load('pipeline_scaler.pkl')
        encoders = joblib.load('encoders.pkl')
    data[numerical_columns] = pipe.transform(data[numerical_columns].to_numpy())
    data_arr_dummies = []
    for col in dummy_columns:
        enc = encoders[col]
        data_arr_dummies.append(enc.transform(data[col].to_numpy().reshape(-1, 1)).toarray())

    data_arr = np.concatenate([data.to_numpy()]+data_arr_dummies, axis=1)

    return data_arr

if __name__ == '__main__':
    train = True
    train_path = 'train.tsv'
    train_data = pd.read_csv(train_path, sep="\t", index_col='id', parse_dates=['release_date'])
    parsed_train_path = "parsed_train_data.pkl"
    if os.path.exists(parsed_train_path):
        parsed_train_data = pd.read_pickle(parsed_train_path)
    else:
        parsed_train_data = parse_data(train_data, train=train)
        # TODO uncomment
        # pd.to_pickle(parsed_train_data, parsed_train_path)

    # TODO - out of memory on competition file?
    print(1)