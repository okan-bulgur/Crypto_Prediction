import pandas as pd
from datetime import timedelta
from ScrapingDatas import coingecko_com_price_data_scraping as pdt
from ScrapingDatas import bitcoin_com_news_urls_scraping as ndt
from ScrapingDatas import data as dt


def getDateInf(crypto):
    path = f'Data Files/{crypto}/csv Files/prices.csv'
    df = pd.read_csv(path)
    startDate = df.iloc[0]['date']
    endDate = df.iloc[-1]['date']

    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)
    return startDate, endDate


def getDateRange(data, startDate, endDate):
    ranges = []
    newStartDate = pd.to_datetime(data.getStartDate())
    newEndDate = pd.to_datetime(data.getEndDate())

    if newStartDate >= endDate or newEndDate > endDate:
        endDate = endDate + timedelta(days=1)
        ranges.append([str(endDate).split()[0], str(newEndDate).split()[0], 1])  # if 2. index == 1 add last of df else add begin

    if newStartDate < startDate or newEndDate <= startDate:
        startDate = startDate - timedelta(days=1)
        ranges.append([str(newStartDate).split()[0], str(startDate).split()[0], 0])

    return ranges


def priceScraping(data, place):
    pdt.createPriceFile(data, place)


def newsScraping(data, place):
    ndt.newsScraping(data, place)


def dataScraping(data):
    startDate, endDate = getDateInf(data.getCryptoType())
    ranges = getDateRange(data, startDate, endDate)
    for inf in ranges:
        newData = dt.Data(inf[0], inf[1], data.getCryptoType())
        print("DataScraping: Start Date: ", inf[0], " End Date: ", inf[1])
        priceScraping(newData, inf[2])
        newsScraping(newData, inf[2])
