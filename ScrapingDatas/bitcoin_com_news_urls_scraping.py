import os
import pandas as pd
import time
import warnings
import tqdm
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from tqdm import tqdm

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', 500)
warnings.filterwarnings("ignore")

#chromedriverPath = 'ScrapingPart/chromedriver/chromedriver.exe'


class newsScraping:
    start_date = None
    end_date = None
    symbol = None
    driver = None
    df = None
    total_news = []

    def __init__(self, data, place):
        pd.set_option('display.max_colwidth', 100)
        plt.show()
        pd.options.plotting.backend = "plotly"

        self.setData(data)
        self.setDriver()
        self.pageAnalysis()
        self.createDataFrame()
        self.createNewsFile(place)

    def setData(self, data):
        self.start_date = data.getStartDate()
        self.end_date = data.getEndDate()
        self.symbol = data.getCryptoType()

    def extract_content(self, driver, count=11):
        products_content_list_to_return = []

        for i in range(1, count):
            item = {}

            # DATE EXTRACTION
            try:
                date_xpath = "/html/body/div[1]/div[3]/div/div[1]/div[2]/div[" + str(i) + "]/div[2]/div[1]/span/time"
                date_element = driver.find_elements(By.XPATH, date_xpath)[0].text
                item['date'] = date_element
                if (pd.to_datetime(date_element, format='%b %d, %Y') > pd.to_datetime(self.end_date,
                                                                                      format='%Y-%m-%d')):
                    continue
                elif (pd.to_datetime(date_element, format='%b %d, %Y') < pd.to_datetime(self.start_date,
                                                                                        format='%Y-%m-%d')):
                    if i == 1:
                        return None
                    break
            except:
                item['date'] = "------"

            # CATEGORY EXTRACTION
            try:
                category_xpath = "/html/body/div[1]/div[3]/div/div[1]/div[2]/div[" + str(i) + "]/div[2]/div[1]/a"
                category_element = driver.find_elements(By.XPATH, category_xpath)[0].text
                item['category'] = category_element
            except:
                item['category'] = "------"

            # TITLE EXTRACTION
            try:
                title_xpath = "/html/body/div[1]/div[3]/div/div[1]/div[2]/div[" + str(i) + "]/div[2]/h3/a"
                title_element = driver.find_elements(By.XPATH, title_xpath)[0].text
                item['title'] = title_element
            except:
                item['title'] = "------"

            # DESCRIPTION EXTRACTION
            try:
                description_xpath = "/html/body/div[1]/div[3]/div/div[1]/div[2]/div[" + str(i) + "]/div[2]/div[2]"
                description_element = driver.find_elements(By.XPATH, description_xpath)[0].text
                item['description'] = description_element
            except:
                item['description'] = "------"

            # URL EXTRACTION
            try:
                url_xpath = "/html/body/div[1]/div[3]/div/div[1]/div[2]/div[" + str(i) + "]/div[2]/h3/a"
                url_element = driver.find_elements(By.XPATH, url_xpath)[0].get_attribute('href')
                item['url'] = url_element
            except:
                item['url'] = "------"

            print(item)
            print('\n')
            print('*****' * 20)
            print('*****' * 20)
            print('\n')

            if item['date'] != "------":
                products_content_list_to_return.append(item)

        return products_content_list_to_return

    def setDriver(self):
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
        self.driver.get(f'https://news.bitcoin.com/?s={self.symbol}')
        time.sleep(3)
        """
        service = ChromeService(executable_path=chromedriverPath)
        self.driver = webdriver.Chrome(service=service)
        self.driver.get(f'https://news.bitcoin.com/?s={self.symbol}')
        time.sleep(3)
        """
    def create_url(self, i=2):
        url = f'https://news.bitcoin.com/page/{str(i)}/?s={self.symbol}'
        return url

    def pageAnalysis(self):
        for i in tqdm(range(1, 2517)):  # 2516
            try:
                print(f'PAGE NUMBER: {i}\n')
                url = self.create_url(i)
                self.driver.get(url)
                time.sleep(5)
                content = self.extract_content(self.driver)
                if content is None:
                    break
                self.total_news.append(content)

            except:
                print("ERROR")
                print("ERROR")
                print("ERROR")
                print("ERROR")

    def createDataFrame(self):
        i = 0
        j = 0
        k = 0

        dictionary_list = []

        for i in range(0, len(self.total_news)):
            for j in range(0, len(self.total_news[i])):
                dictionary_list.append(self.total_news[i][j])

        self.df = pd.DataFrame.from_records(dictionary_list,
                                            columns=['date', 'url', 'title', 'description', 'category'])
        self.df['date'] = pd.to_datetime(self.df['date'], format='%b %d, %Y')

    def createNewsFile(self, place):
        pathXlsx = f'Data Files/{self.symbol}/xlsx Files/bitcoin.com_news_url.xlsx'
        pathCsv = f'Data Files/{self.symbol}/csv Files/bitcoin.com_news_url.csv'

        if not os.path.exists(pathCsv):
            self.df.to_csv(pathCsv, index=False, encoding='utf-16')
            self.df.to_excel(pathXlsx, index=False)
            return

        news_df = pd.read_csv(pathCsv, encoding='utf-16')
        news_df.reset_index(drop=True, inplace=True)

        self.df['date'] = pd.to_datetime(self.df['date']).dt.date

        if place == 0:
            news_df = pd.concat([news_df, self.df], ignore_index=True)
        else:
            news_df = pd.concat([self.df, news_df], ignore_index=True)

        news_df.to_csv(pathCsv, index=False, encoding='utf-16')
        news_df.to_excel(pathXlsx, index=False)
