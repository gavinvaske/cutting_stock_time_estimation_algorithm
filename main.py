import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neural_network import MLPClassifier

COLUMN_DIE_NUMBER = 'DIE NUMBER'
COLUMN_TOTAL_REPEAT = 'Total Repeat'
COLUMN_TOTAL_ACROSS = 'Total Across'
COLUMN_FINISH = 'Finish'
COLUMN_LENGTH_FEET = 'Length FT'
COLUMN_STATUS = 'Status'
COLUMN_FEET_PER_MINUTE = 'Feet/Min'

def normalize(data_frame, column_name):
    return MinMaxScaler().fit_transform(np.array(data_frame[column_name]).reshape(-1, 1))

def normalize_columns(dataframe, column_names):
    for column_name in column_names:
        dataframe[column_name] = normalize(dataframe, column_name)
    return dataframe

def fill_na_values_in_column(dataframe, column_name, new_value):
    dataframe[column_name] = dataframe[column_name].fillna(new_value)
    return dataframe

def one_hot_encode_columns(dataframe, column_names):
    for column_name in column_names:
        dataframe[column_name] = dataframe[column_name].str.upper()
        one_hot = pd.get_dummies(dataframe[column_name], prefix=column_name)
        dataframe = dataframe.drop(column_name, axis=1)
        dataframe = dataframe.join(one_hot)
    return dataframe

if __name__ == '__main__':
    csv = pd.read_csv('./input_data/press_logs.csv', usecols=[
        COLUMN_DIE_NUMBER,
        COLUMN_TOTAL_REPEAT,
        COLUMN_TOTAL_ACROSS,
        COLUMN_FEET_PER_MINUTE,
        COLUMN_LENGTH_FEET,
        COLUMN_FINISH,
        COLUMN_STATUS
    ])

    print('Shape BEFORE dropping NA rows: ', csv.shape)

    csv = csv.dropna(
        subset=[
            COLUMN_TOTAL_REPEAT,
            COLUMN_TOTAL_ACROSS,
            COLUMN_FEET_PER_MINUTE
        ]
    )

    print('Shape AFTER dropping NA rows: ', csv.shape)

    csv = normalize_columns(
        csv,
        [
            COLUMN_TOTAL_REPEAT,
            COLUMN_TOTAL_ACROSS
        ]
    )

    csv = fill_na_values_in_column(csv, COLUMN_FINISH, 'NO_FINISH')
    csv = fill_na_values_in_column(csv, COLUMN_STATUS, 'NO_STATUS')

    csv = one_hot_encode_columns(
        csv,
        [
            COLUMN_FINISH,
            COLUMN_STATUS
        ]
    )

    csv.to_csv('./output_data/processed_press_logs.csv', index=False)

    print('CSV statistics: ', csv.describe())


