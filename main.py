import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

COLUMN_DIE_NUMBER = 'DIE NUMBER'
COLUMN_TOTAL_REPEAT = 'Total Repeat'
COLUMN_TOTAL_ACROSS = 'Total Across'
COLUMN_FINISH = 'Finish'
COLUMN_LENGTH_FEET = 'Length FT'
COLUMN_STATUS = 'Status'
COLUMN_FEET_PER_MINUTE = 'Feet/Min'

MINIMUM_JOB_LENGTH_ACCEPTED = 400

FINISH_TYPE_NO_FINISH = 'NO_FINISH'
FINISH_TYPE_UV = 'UV'
FINISH_TYPE_LAMINATION = 'LAMINATION'
TARGET_VARIABLE_GROUP_SIZE = 30

def normalize(data_frame, column_name):
    return MinMaxScaler().fit_transform(np.array(data_frame[column_name]).reshape(-1, 1))

def normalize_columns(dataframe, column_names):
    for column_name in column_names:
        dataframe[column_name] = normalize(dataframe, column_name)
    return dataframe

def de_normalize(normalized_value, max, min):
    return (normalized_value * (max - min) + min);

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
            COLUMN_FEET_PER_MINUTE,
            COLUMN_LENGTH_FEET
        ]
    )
    csv[COLUMN_FEET_PER_MINUTE] = csv[COLUMN_FEET_PER_MINUTE].apply(lambda x: round(x / TARGET_VARIABLE_GROUP_SIZE) * TARGET_VARIABLE_GROUP_SIZE)

    print('Shape AFTER dropping NA rows: ', csv.shape)

    # Remove commas from columns, (Ex: 2,321 is interpreted as a string until the comma is removed)
    csv = csv.replace(',', '', regex=True)
    csv[COLUMN_LENGTH_FEET] = pd.to_numeric(csv[COLUMN_LENGTH_FEET])

    # Drop rows whose "LENGTH FT" is too small (i.e. small sample size) to avoid confusing the neural network during training
    csv.drop(csv[csv[COLUMN_LENGTH_FEET] < MINIMUM_JOB_LENGTH_ACCEPTED].index, inplace=True)

    csv = normalize_columns(
        csv,
        [
            COLUMN_TOTAL_REPEAT,
            COLUMN_TOTAL_ACROSS
        ]
    )

    # Uppercase all values before executing regex expressions
    csv[COLUMN_FINISH] = csv[COLUMN_FINISH].str.upper()
    csv[COLUMN_FINISH] = csv[COLUMN_FINISH].replace(regex={
        r'(-){2,}': FINISH_TYPE_NO_FINISH,      # Replace any finish containing at least "--" to "NO_FINISH"
        r'.*(UV).*': FINISH_TYPE_UV,            # Replace any *UV* Finish types with "UV"
        r'.*(\d).*': FINISH_TYPE_LAMINATION     # Replace any Finish type containing a number with "LAMINATION"
    })

    csv = fill_na_values_in_column(csv, COLUMN_FINISH, FINISH_TYPE_NO_FINISH)
    csv = fill_na_values_in_column(csv, COLUMN_STATUS, 'NO_STATUS')

    csv = one_hot_encode_columns(
        csv,
        [
            COLUMN_FINISH,
            COLUMN_STATUS
        ]
    )
    print('Shape BEFORE removing small jobs: ', csv.shape)

    print('Shape AFTER removing small jobs: ', csv.shape)

    csv.to_csv('./output_data/processed_press_logs.csv', index=False)

    print('CSV statistics (after normailization): ', csv.describe())

    y = csv[COLUMN_FEET_PER_MINUTE]
    X = csv.drop([COLUMN_FEET_PER_MINUTE, COLUMN_DIE_NUMBER, COLUMN_LENGTH_FEET], axis=1)

    print('\nPrinting Y shape: ', y.shape)
    print('\nPrinting X shape: ', X.shape)
    print('\nPrinting X columns: ', X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.15)

    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6), max_iter=5000).fit(X_train, y_train)
    # clf = MLPRegressor(solver='lbfgs', random_state=1, max_iter=5000).fit(X_train, y_train)

    # Expect: 55
    guessIt = np.array([0.05725190839694656,0.9494949494949496, True, False, False, False, False, False, True, False])
    result = clf.predict(guessIt.reshape(1, -1))
    print('result: ', result)
    mlp_prediction_score = clf.score(X_test, y_test)

    print('Resulting score: ', mlp_prediction_score)


