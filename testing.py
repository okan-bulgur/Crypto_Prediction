import pandas as pd
from datetime import datetime
from Prediction import prediction
import pyttsx3
from tkinter import END


movementPred = None

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def getInformations(testDatas_df, index):
    crypto = testDatas_df.loc[index, 'Crypto']
    trainStartDate = testDatas_df.loc[index, 'Train Start Date']
    trainEndDate = testDatas_df.loc[index, 'Train End Date']
    testStartDate = testDatas_df.loc[index, 'Test Start Date']
    testEndDate = testDatas_df.loc[index, 'Test End Date']

    inf = {
        'Crypto': crypto,
        'Train Start Date': trainStartDate,
        'Train End Date': trainEndDate,
        'Test Start Date': testStartDate,
        'Test End Date': testEndDate
    }

    return inf


def getResult(df, index, inf):
    testStartDate = inf['Test Start Date']
    testEndDate = inf['Test End Date']

    testStartDate = datetime.strptime(testStartDate, "%Y-%m-%d")
    testEndDate = datetime.strptime(testEndDate, "%Y-%m-%d")

    data_df = pd.read_csv('data.csv')
    data_df.set_index('date', inplace=True)

    testStartDate = str(testStartDate).split(" ")[0]
    testEndDate = str(testEndDate).split(" ")[0]

    resultData = data_df.loc[testStartDate: testEndDate]
    resultList = resultData['movement'].to_list()
    resultStr = ','.join(map(str, resultList))

    df.loc[index, 'Result'] = resultStr

    return df


def getPredict(df, index, inf):
    global movementPred
    """exec(open('Prediction/price_prediction_classification_short.py').read(),
         {'symbol': inf['Crypto'], 'showPlot': False,
          'train_start_date': inf['Train Start Date'], 'train_end_date': inf['Train End Date'],
          'test_start_date': inf['Test Start Date'], 'test_end_date': inf['Test End Date']
          })

    movementPred = rst.result"""

    model = 'gbc_model'
    data = prediction.startPrepareData(inf['Crypto'])
    x_train, x_test, y_train, y_test = prediction.splitDataByDate \
        (data, inf['Train Start Date'],
         inf['Train End Date'], inf['Test Start Date'],
         inf['Test End Date'])
    prediction.startModel(data, model, x_train, x_test, y_train, y_test)

    test_data_index = data.loc[inf['Test Start Date']: inf['Test End Date']].index

    print("test data index: ", test_data_index)

    movementPred = prediction.startPrediction(model, data, x_train, x_test, y_train, y_test, test_data_index)

    predicted = [value for value in movementPred.values()]
    predictedStr = ','.join(map(str, predicted))
    df.loc[index, 'Predicted'] = predictedStr

    return df


def calculateConsistency(df, index):
    predicted = df.loc[index, 'Predicted']
    result = df.loc[index, 'Result']

    predicted = predicted.split(',')
    result = result.split(',')

    numerator = 0
    denominator = 0

    for i in range(0, len(predicted)):
        if predicted[i] == result[i]:
            numerator += 1
        denominator += 1

    consistency = str(round(numerator * 100 / denominator, 2)) + '%'
    df.loc[index, 'Consistency'] = consistency

    return df


def calculateAvgOfConsistency(startIndex, endIndex, txtArea):
    df = pd.read_csv('testDatas.csv')
    sum = 0
    totalData = 0
    for index in range(startIndex - 2, endIndex - 1):
        cons = df.loc[index, 'Consistency']

        if str(cons) == "nan":
            print(f'Index: {index + 2} is not calculated.')
            continue

        cons = float(cons.split("%")[0])
        sum += cons
        totalData += 1

    try:
        average = round(sum / totalData, 2)
    except ZeroDivisionError:
        print(f'\033[33mAverage cannot be calculated\033[0m\n')
        speak(f'Average cannot be calculated')
        return

    print(f'Sum of Consistency: {sum}')
    print(f'Number of Data: {totalData}')
    print(f'Average: {average}')

    speak(f'Average is {average}')

    txtArea.config(state='normal')
    txtArea.delete('1.0', END)
    txt = f'In index {startIndex} to {endIndex} : average of consistency is {average}'
    txtArea.insert('end', txt)
    txtArea.config(state='disable')


def automation(startIndex, endIndex):
    testDatas_df = pd.read_csv('testDatas.csv')

    if endIndex > testDatas_df.shape[0]:
        endIndex = testDatas_df.shape[0] + 1

    for index in range(startIndex - 2, endIndex - 1):
        speak(f'{index + 2} is started')
        inf = getInformations(testDatas_df, index)
        print(f'\033[32mIndex : {index + 2}\nInf : {inf}\033[0m\n')
        testDatas_df = getResult(testDatas_df, index, inf)
        testDatas_df = getPredict(testDatas_df, index, inf)
        testDatas_df = calculateConsistency(testDatas_df, index)
        testDatas_df.to_csv('testDatas.csv', index=False)
        speak(f'{index + 2} is ended')

    speak('Test automation is finished')


def manuel(inf):
    global movementPred

    testDatas_df = pd.read_csv('testDatas.csv')

    index = testDatas_df.shape[0] + 1

    print(f'\033[32mIndex : {index + 2}\nInf : {inf}\033[0m\n')

    testDatas_df = getResult(testDatas_df, index, inf)
    testDatas_df = getPredict(testDatas_df, index, inf)
    testDatas_df = calculateConsistency(testDatas_df, index)
    testDatas_df.to_csv('testDatas.csv', index=False)

