import pandas as pd

result = None


def createResultFile(symbol):
    global result

    date = list(result.keys())
    predicted = list(result.values())

    df = pd.DataFrame({'Date': date, 'Predicted': predicted})

    df.to_excel(f'Files/{symbol}/xlsx Files/result.xlsx', index=False)
    df.to_csv(f'Files/{symbol}/csv Files/result.csv', index=False, encoding='utf-16')
