from ScrapingDatas import coingecko_com_price_data_scraping as pdt
from ScrapingDatas import bitcoin_com_news_urls_scraping as ndt


def priceScraping(data):
    pdt.createPriceFile(data)


def newsScraping(data):
    ndt.newsScraping(data)


def dataScraping(data):
    priceScraping(data)
    newsScraping(data)
