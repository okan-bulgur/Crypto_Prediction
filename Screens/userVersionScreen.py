import datetime
from tkinter import *
from datetime import timedelta
import pandas as pd
import tkinter.messagebox
from ScrapingDatas import data
from ScrapingDatas import dataScraping as ds
from Screens import screenElements as se
import os

import testing as ta

title = "User Version"
screenHeight = 700
screenWidth = 500


class UserScreen:
    cryptoList = ["bitcoin", "ethereum", "solana", "cardano"]
    model = 'gbc_model'
    startDateForData = "2020-01-01"
    crypto = StringVar
    cryptoLabel = Label
    calculateLabel = Label

    # Frames
    generalFrame = Frame
    inputFrame = Frame
    outputFrame = Frame

    resultLabel = Label
    txtArea = Text

    def __init__(self, frame):
        self.window = frame
        self.crtScreen()

    def crtScreen(self):
        self.window.config(background=se.backgroundColor1)

        header = Label(self.window, text="Crypto Prediction", font=(se.fontType, 40, 'bold'), fg=se.foregroundColor,
                       bg=se.backgroundColor1)
        header.pack(side=TOP)

        self.crtGeneralFrame()

    def crtGeneralFrame(self):
        relHeight = 0.8
        relWidth = 1
        inputFrameRelHeight = 0.7
        outputFrameRelHeight = 1 - inputFrameRelHeight

        self.generalFrame = Frame(self.window, bg=se.backgroundColor2)
        self.generalFrame.place(relx=1 - relWidth, rely=1 - relHeight, relwidth=relWidth, relheight=relHeight)

        self.crtInputFrame(0.2, 0.1, 1, inputFrameRelHeight, se.backgroundColor2)  # relx, rely, relwidth, relheight, bg
        self.crtOutputFrame(0, inputFrameRelHeight, 1, outputFrameRelHeight, se.backgroundColor1)

    def crtInputFrame(self, relx, rely, relwidth, relheight, bg):
        self.inputFrame = Frame(self.generalFrame, bg=bg)
        self.inputFrame.place(relx=relx, rely=rely, relwidth=relwidth, relheight=relheight)

        self.crtInputElements(self.inputFrame, bg)

    def crtOutputFrame(self, relx, rely, relwidth, relheight, bg):
        self.outputFrame = Frame(self.generalFrame, bg=bg)
        self.outputFrame.place(relx=relx, rely=rely, relwidth=relwidth, relheight=relheight)

        self.resultLabel = se.crtLabel(self.outputFrame, {'row': 0, 'column': 0, 'padx': 0, 'pady': 0})

        self.txtArea = se.crtOutputTextArea(self.outputFrame, {'relx': 0, 'rely': 0, 'relheight': 1, 'relwidth': 1})

    def crtInputElements(self, frame, bg):
        # header
        headerTxt = "Predict Tomorrow's\nMovement"
        header = Label(frame, text=headerTxt, font=(se.fontType, 20, 'bold'), fg=se.foregroundColor, bg=bg)
        header.grid(row=0, column=0)

        # crypto
        cryptoBtn = se.crtMenuBtn(frame, "Crypto", {'row': 1, 'column': 0, 'padx': 0, 'pady': 10},
                                  self.cryptoList, self.setCryptoType, None)
        self.cryptoLabel = se.crtLabel(frame, {'row': 2, 'column': 0, 'padx': 0, 'pady': 10})

        # calculate
        scrapeDataBtn = se.crtBtn(frame, "Calculate", {'row': 3, 'column': 0, 'padx': 0, 'pady': 10})
        scrapeDataBtn.config(command=self.calculate)
        self.calculateLabel = se.crtLabel(frame, {'row': 4, 'column': 0, 'padx': 0, 'pady': 10})
        self.calculateLabel.config(fg=se.finishForegroundColor)

    def setCryptoType(self, crypto, type):
        if crypto is not None:
            self.crypto = crypto
            self.cryptoLabel.config(text=crypto)

            if not os.path.exists(f'Data Files/{crypto}/csv Files'):
                os.makedirs(f'Data Files/{crypto}/csv Files')
                os.makedirs(f'Data Files/{crypto}/xlsx Files')
        else:
            tkinter.messagebox.showwarning(title="Error", message="Choose Crypto Type")

    def calculate(self):
        today = str(datetime.date.today())
        todayDateTime = pd.to_datetime(today)

        trainEndDate = todayDateTime - timedelta(days=2)
        testDate = todayDateTime - timedelta(days=1)

        trainEndDate = str(trainEndDate).split()[0]
        testDate = str(testDate).split()[0]

        inf = {
            'Crypto': self.crypto,
            'Model': self.model,
            'Train Start Date': self.startDateForData,
            'Train End Date': trainEndDate,
            'Test Start Date': testDate,
            'Test End Date': testDate
        }

        # data scraping
        dataScraping(inf['Train Start Date'], inf['Test End Date'], self.crypto)
        # prepare data
        prepareData(self.crypto)
        # prediction
        self.prediction(inf)

    def prediction(self, inf):
        print("Start Prediction")
        self.txtArea.config(state='normal')
        self.txtArea.delete('1.0', END)

        ta.manuel(inf)

        movementPred = ta.movementPred

        for date, predicted in movementPred.items():
            result = "It is predicted to rise" if predicted == 1 else "It is predicted to fall"
            date = str(date).split(" ")[0]
            txt = f"{date}: {result}\n"
            self.txtArea.insert('end', txt)

        self.txtArea.config(state='disable')
        print("End Prediction")


def dataScraping(startDate, endDate, crypto):
    print("Start Data Scraping")
    newData = data.Data(startDate, endDate, crypto)
    ds.dataScraping(newData)
    print("End Data Scraping")


def prepareData(crypto):
    print("Start Prepare Data")
    exec(open('PrepareDatas/prepare_data.py').read(), {'symbol': crypto})
    print("End Prepare Data")
