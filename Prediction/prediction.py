import pandas as pd
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from PrepareDatas import prepare_last_data as pld


def splitDataByDate(data, start_train_date, end_train_date, start_test_date, end_test_date):
    start_train_date = datetime.strptime(start_train_date, "%Y-%m-%d")
    end_train_date = datetime.strptime(end_train_date, "%Y-%m-%d")

    start_test_date = datetime.strptime(start_test_date, "%Y-%m-%d")
    end_test_date = datetime.strptime(end_test_date, "%Y-%m-%d")

    train_data = data.loc[start_train_date: end_train_date]
    test_data = data.loc[start_test_date: end_test_date]

    drop_column = 'movement'
    target = 'movement'

    x_train = train_data.drop([drop_column], axis=1)
    x_test = test_data.drop([drop_column], axis=1)
    y_train = train_data[target]
    y_test = test_data[target]

    return x_train, x_test, y_train, y_test


def splitData(data, testSize):
    drop_column = 'movement'
    target = 'movement'

    x = data.drop([drop_column], axis=1)
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, random_state=101)

    return x_train, x_test, y_train, y_test


def deployModel(model, path):
    joblib.dump(model, path)


def loadModel(path):
    model = joblib.load(path)
    return model


def getReports(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f'Confusion Matrix:\n{cm}\n\nClassification Report:\n{cr}\n')


def gbcModel(modelPath, x_train, x_test, y_train, y_test):
    print("\t\t\t*** GBC MODEL ***\n")

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    # Create the GradientBoostingClassifier
    gbm = GradientBoostingClassifier()

    # Perform Grid Search
    grid_search = GridSearchCV(gbm, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Access the results
    results = grid_search.cv_results_

    # Display the progress
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f'Mean Score: {mean_score}\nParameters: {params}\n')

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'\nBest Score: {best_score}\n')

    # Train the model with the best parameters
    best_gbm = GradientBoostingClassifier(**best_params)
    best_gbm.fit(x_train, y_train)

    # Evaluate the model
    test_score = best_gbm.score(x_test, y_test)

    print("Best Parameters: ", best_params)
    print("Best Score: ", best_score)
    print("Test Score with Best Parameters: ", test_score)

    y_pred = best_gbm.predict(x_test)

    getReports(y_test, y_pred)

    probs = best_gbm.predict_proba(x_test)
    probs = probs[:, 1]

    auc = roc_auc_score(y_test, probs)
    print(f'\nAUC: {auc * 100}\n')

    deployModel(best_gbm, modelPath)


def startModel(data, model, x_train, x_test, y_train, y_test):
    crypto = data.columns[0].split('_')[0]

    if model == 'gbc_model':
        modelPath = f'{crypto}/{model}.pkl'
        gbcModel(modelPath, x_train, x_test, y_train, y_test)


def startPrepareData(crypto):
    dataXlsxPath = f'Data Files/{crypto}/xlsx Files/final_data_TEST_DATA.xlsx'
    finalDataCsvPath = f'Data Files/{crypto}/csv Files/final_data.csv'
    finalDataXlsxPath = f'Data Files/{crypto}/xlsx Files/final_data.xlsx'

    data = pd.read_excel(dataXlsxPath)
    data.set_index('date', inplace=True)

    data = pld.generateData(data, crypto)

    data.to_csv(finalDataCsvPath)
    data.to_excel(finalDataXlsxPath)

    return data


def startPrediction(model, data, x_train, x_test, y_train, y_test, test_data_index):
    predicted = {}

    crypto = data.columns[0].split('_')[0]
    modelPath = f'{crypto}/{model}.pkl'
    model = loadModel(modelPath)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    getReports(y_test, y_pred)

    for date, move in zip(test_data_index, y_pred):
        predicted[date] = move

    print("Predicted Movement is : ", predicted)

    return predicted
