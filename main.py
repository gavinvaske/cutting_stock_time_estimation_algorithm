import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import r_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt
import seaborn as sns

COLUMN_DIE_NUMBER = 'DIE NUMBER'
COLUMN_TOTAL_REPEAT = 'Total Repeat'
COLUMN_TOTAL_ACROSS = 'Total Across'
COLUMN_FINISH = 'Finish'
COLUMN_LENGTH_FEET = 'Length FT'
COLUMN_STATUS = 'Status'
COLUMN_FEET_PER_MINUTE = 'Feet/Min'

MINIMUM_JOB_LENGTH_ACCEPTED = 1

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

def print_pearson_correlation_coefficients(X, y):
    pearson_correlations = r_regression(X.to_numpy(), y.to_numpy())
    n_attributes = X.columns.size

    print('\n')
    for x in range(n_attributes):
        print('Column: ', X.columns[x], ': Pearson Correlation: ', pearson_correlations[x])
    print('\n')

def remove_outliers(dataframe):
    lower_limit_ft_per_min = 1
    lower_limit_length = 1

    print('shape BEFORE removing outliers: ', dataframe.shape)
    dataframe = dataframe[(dataframe[COLUMN_FEET_PER_MINUTE] > lower_limit_ft_per_min)]
    dataframe = dataframe[(dataframe[COLUMN_LENGTH_FEET] > lower_limit_length)]
    print('shape AFTER removing outliers: ', dataframe.shape)

    return dataframe

def clean_data(dataframe):
    print('Shape BEFORE dropping NA rows: ', dataframe.shape)
    dataframe = dataframe.dropna(
        subset=[
            COLUMN_TOTAL_REPEAT,
            COLUMN_TOTAL_ACROSS,
            COLUMN_FEET_PER_MINUTE
        ]
    )
    print('Shape AFTER dropping NA rows: ', dataframe.shape)

    # Remove commas from columns, (Ex: 2,321 is interpreted as a string until the comma is removed)
    dataframe = dataframe.replace(',', '', regex=True)
    dataframe[COLUMN_LENGTH_FEET] = pd.to_numeric(dataframe[COLUMN_LENGTH_FEET])

    dataframe = remove_outliers(dataframe)

    dataframe = normalize_columns(
        dataframe,
        [
            COLUMN_TOTAL_REPEAT,
            COLUMN_TOTAL_ACROSS,
            COLUMN_LENGTH_FEET
        ]
    )

    # # Uppercase all values before executing regex expressions
    dataframe[COLUMN_FINISH] = dataframe[COLUMN_FINISH].str.upper()
    dataframe[COLUMN_FINISH] = dataframe[COLUMN_FINISH].replace(regex={
        r'(-){2,}': FINISH_TYPE_NO_FINISH,      # Replace any finish containing at least "--" to "NO_FINISH"
        r'.*(UV).*': FINISH_TYPE_UV,            # Replace any *UV* Finish types with "UV"
        r'.*(\d).*': FINISH_TYPE_LAMINATION     # Replace any Finish type containing a number with "LAMINATION"
    })

    dataframe = fill_na_values_in_column(dataframe, COLUMN_FINISH, FINISH_TYPE_NO_FINISH)
    dataframe = fill_na_values_in_column(dataframe, COLUMN_STATUS, 'NO_STATUS')

    dataframe = one_hot_encode_columns(
        dataframe,
        [
            COLUMN_FINISH,
            COLUMN_STATUS
        ]
    )

    return dataframe

if __name__ == '__main__':
    csv = pd.read_csv('input_data/press_logs_with_status.csv', usecols=[
        COLUMN_DIE_NUMBER,
        COLUMN_TOTAL_REPEAT,
        COLUMN_TOTAL_ACROSS,
        COLUMN_FEET_PER_MINUTE,
        COLUMN_LENGTH_FEET,
        COLUMN_FINISH,
        COLUMN_STATUS
    ])

    csv = clean_data(csv)

    ## PRINTING STATISTICS about dataset
    print(csv[COLUMN_FEET_PER_MINUTE].describe())

    plt.hist(csv[COLUMN_FEET_PER_MINUTE], bins=20)
    plt.xlabel('Feet per Second')
    plt.ylabel('Count')
    plt.show()

    cor = csv.drop([COLUMN_DIE_NUMBER], axis=1).corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    csv.to_csv('./output_data/processed_press_logs.csv', index=False)

    print('CSV statistics (after normailization): ', csv.describe())

    y = csv[COLUMN_FEET_PER_MINUTE]
    X = csv[[COLUMN_TOTAL_ACROSS, COLUMN_LENGTH_FEET, 'Status_NO_STATUS']]   # Select Columns Highly correlatted to target variable but not highly correlated to one another

    print('\nPrinting Y shape: ', y.shape)
    print('\nPrinting X shape: ', X.shape)
    print('\nPrinting X columns: ', X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.25)


    print_pearson_correlation_coefficients(X, y)

    # Dummy Regressor (a baseline-model that always predicts the mean of the target variable)
    dummy_regressor = DummyRegressor(strategy='constant', constant=73).fit(X_train, y_train)
    dummy_regressor_score = dummy_regressor.score(X_test, y_test)

    # Ridge Linear Regression
    ridge_linear_regression = linear_model.Ridge(alpha=0.5, positive=True).fit(X_train, y_train)
    ridge_linear_regression_score = ridge_linear_regression.score(X_test, y_test)

    # Linear Regression
    linear_regression = linear_model.LinearRegression(positive=True).fit(X_train, y_train)
    linear_regression_score = linear_regression.score(X_test, y_test)

    # MLP Regressor
    mlp_regressor = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(5), max_iter=5000).fit(X_train, y_train)
    mlp_regressor_score = mlp_regressor.score(X_test, y_test)

    # Lasso Regression
    lasso_regression = linear_model.Lasso(alpha=0.5, positive=True).fit(X_train, y_train)
    lasso_regression_score = lasso_regression.score(X_test, y_test)

    print('#### R^2 SCORES ####')
    print('Dummy Regressor (baseline algorithm) R^2 score = ', dummy_regressor_score)
    print('Ridge R^2 score = ', ridge_linear_regression_score)
    print('Linear-Regression R^2 score = ', linear_regression_score)
    print('MLP R^2 score = ', mlp_regressor_score)
    print('Lasso R^2 score = ', lasso_regression_score)

    print('\n#### Algorithm Predictions ####')

    sample = X_test[0].reshape(1, -1)
    correct_prediction = y_test[0]
    print('About to predict: ', sample, '\nHoping to get a value of: ', correct_prediction)
    print('\n')
    print('Dummy-Regressor Guess: ', dummy_regressor.predict(sample))
    print('Ridge Guess: ', ridge_linear_regression.predict(sample))
    print('Linear-Regression Guess: ', linear_regression.predict(sample))
    print('MLP Guess: ', mlp_regressor.predict(sample))
    print('Lasso Guess: ', lasso_regression.predict(sample))


    # Predict all rows of the original dataset and output results to a CSV file
    dummy_regressor_predictions = dummy_regressor.predict(X.to_numpy())
    ridge_linear_regression_predictions = ridge_linear_regression.predict(X.to_numpy())
    linear_regression_predictions = linear_regression.predict(X.to_numpy())
    mlp_regressor_predictions = mlp_regressor.predict(X.to_numpy())
    lasso_regression_predictions = lasso_regression.predict(X.to_numpy())

    predictions = {
        'Die Number': csv[COLUMN_DIE_NUMBER],
        'Actual ft/min': csv[COLUMN_FEET_PER_MINUTE],
        'Storms Dummy Guess': dummy_regressor_predictions,
        'Ridge Linear Regression': ridge_linear_regression_predictions,
        'Linear Regression': linear_regression_predictions,
        'Neural Network': mlp_regressor_predictions,
        'Lasso Regression': lasso_regression_predictions
    }

    pd.DataFrame(predictions).to_csv('./output_data/predictions.csv', index=None, )


