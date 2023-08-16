from datetime import timedelta, datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PrepareDatas import prepare_last_data as pld
from Prediction import models


def splitDataByDate(data, start_train_date, end_train_date, start_test_date, end_test_date):
    """
    start_train_date = datetime.strptime(start_train_date, "%Y-%m-%d")
    end_train_date = datetime.strptime(end_train_date, "%Y-%m-%d")

    start_test_date = datetime.strptime(start_test_date, "%Y-%m-%d")
    end_test_date = datetime.strptime(end_test_date, "%Y-%m-%d")
    """

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


def startModel(data, model, x_train, x_test, y_train, y_test):
    crypto = data.columns[0].split('_')[0]

    if model == 'gbc_model':
        modelPath = f'Models/{crypto}/{model}.pkl'
        models.gbcModel(modelPath, x_train, x_test, y_train, y_test)


def startPrepareData(crypto):
    dataCsvPath = f'Data Files/{crypto}/csv Files/final_data_TEST_DATA.csv'
    finalDataCsvPath = f'Data Files/{crypto}/csv Files/final_data.csv'
    finalDataXlsxPath = f'Data Files/{crypto}/xlsx Files/final_data.xlsx'

    data = pd.read_csv(dataCsvPath)
    data.set_index('date', inplace=True)

    data = pld.generateData(data, crypto)

    data.to_csv(finalDataCsvPath)
    data.to_excel(finalDataXlsxPath)

    return data


def startPrediction(model, data, x_train, x_test, y_train, y_test, test_data_index):
    predicted = {}

    crypto = data.columns[0].split('_')[0]
    modelPath = f'Models/{crypto}/{model}.pkl'
    model = models.loadModel(modelPath)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    models.getReports(y_test, y_pred)

    for date, move in zip(test_data_index, y_pred):
        date = datetime.strptime(date, "%Y-%m-%d")
        date = date + timedelta(days=1)
        predicted[str(date).split(" ")[0]] = move

    print("Predicted Movement is : ", predicted)

    return predicted
