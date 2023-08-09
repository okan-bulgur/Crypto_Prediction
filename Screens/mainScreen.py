from tkinter import *
from tkcalendar import Calendar
from functools import partial
import tkinter.messagebox
from ScrapingDatas import data
from ScrapingDatas import dataScraping as ds
import os

import testing as ta

screenHeight = 1000
screenWidth = 1500
fontType = 'Arial'
fontSize = 20
backgroundColor1 = '#D2E9E9'
backgroundColor2 = '#F8F6F4'
foregroundColor = '#27374D'
finishForegroundColor = '#16FF00'

btnForegroundColor = '#27374D'
btnBackgroundColor = '#C4DFDF'
btnActiveForegroundColor = '#27374D'
btnActiveBackgroundColor = '#D2E9E9'
btnFontSize = 12
btnFontType = 'Arial'


class MainScreen:
    cryptoList = ["bitcoin", "ethereum", "solana", "cardano"]

    # Frames
    generalFrame = Frame
    inputFrame = Frame
    outputFrame = Frame

    # Parts
    dataPart = Frame
    robertaPart = Frame
    prepareDataPart = Frame
    predictPart = Frame
    testDataPart = Frame

    # Data Part Inputs
    dataScpStartDate = None
    dataScpEndDate = None
    dataScpCrypto = None

    # Roberta Part Inputs
    robertaCrypto = None

    # Prepare Data Part Inputs
    dataCrypto = None

    # Predict Part Inputs
    predictCrypto = None
    trainDataStartDate = None
    trainDataEndDate = None
    testDataStartDate = None
    testDataEndDate = None

    # Test Data Part Inputs
    startIndex = None
    endIndex = None

    # Data Part Labels
    dataScpLabel = Label
    dataScpStartDateLabel = Label
    dataScpEndDateLabel = Label
    dataScpCryptoLabel = Label
    # Roberta Part Labels
    robertaCryptoLabel = Label
    robertaTrainLabel = Label
    # Prepare Data Part Labels
    dataCryptoLabel = Label
    dataPrepareLabel = Label
    # Predict Part Labels
    predictCryptoLabel = Label
    trainDataStartDateLabel = Label
    trainDataEndDateLabel = Label
    testDataStartDateLabel = Label
    testDataEndDateLabel = Label
    predictLabel = Label
    # Test Data Part Labels
    startIndexLabel = Label
    endIndexLabel = Label
    testDatasPredictLabel = Label
    testDatasAvgLabel = Label
    # Result Part Labels
    resultLabel = Label
    # output Part
    txtArea = Text

    inputParts = {
        'Data Part': dataPart,
        'Roberta Part': robertaPart,
        'Prepare Data Part': prepareDataPart,
        'Predict Part': predictPart,
        'Test Data Part': testDataPart
    }

    inputs = {
        # Data Part Inputs
        'dataScpStartDate': [dataScpStartDate, dataScpStartDateLabel],
        'dataScpEndDate': [dataScpEndDate, dataScpEndDateLabel],
        'dataScpCrypto': [dataScpCrypto, dataScpCryptoLabel],
        # Roberta Part Inputs
        'robertaCrypto': [robertaCrypto, robertaCryptoLabel],
        # Prepare Data Parts
        'dataCrypto': [dataCrypto, dataCryptoLabel],
        # Predict Part Inputs
        'predictCrypto': [predictCrypto, predictCryptoLabel],
        'trainDataStartDate': [trainDataStartDate, trainDataStartDateLabel],
        'trainDataEndDate': [trainDataEndDate, trainDataEndDateLabel],
        'testDataStartDate': [testDataStartDate, testDataStartDateLabel],
        'testDataEndDate': [testDataEndDate, testDataEndDateLabel],
        # Test Data Part Inputs
        'startIndex': [startIndex, startIndexLabel],
        'endIndex': [endIndex, endIndexLabel]
    }

    def __init__(self):
        self.window = Tk()
        self.robertaShowPlot = BooleanVar()
        self.dataShowPlot = BooleanVar()
        self.predictShowPlot = BooleanVar()
        self.inputs['robertaShowPlot'] = self.robertaShowPlot
        self.inputs['dataShowPlot'] = self.dataShowPlot
        self.inputs['predictShowPlot'] = self.predictShowPlot
        self.crtScreen()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def crtScreen(self):
        self.window.title("Crypto Prediction")
        self.window.geometry(f"{screenWidth}x{screenHeight}")
        self.window.config(background=backgroundColor1)

        icon = PhotoImage(file='res/btcIcon.png')
        self.window.iconphoto(True, icon)

        header = Label(self.window, text="Crypto Prediction", font=(fontType, 40, 'bold'), fg=foregroundColor,
                       bg=backgroundColor1)
        header.pack(side=TOP)

        self.crtGeneralFrame()

    def crtGeneralFrame(self):
        relHeight = 0.8
        relWidth = 1
        inputFrameRelHeight = 0.7
        outputFrameRelHeight = 1 - inputFrameRelHeight

        self.generalFrame = Frame(self.window)
        self.generalFrame.place(relx=1 - relWidth, rely=1 - relHeight, relwidth=relWidth, relheight=relHeight)

        self.crtInputFrame(0, 0, 1, inputFrameRelHeight, backgroundColor2)  # relx, rely, relwidth, relheight, bg
        self.crtOutputFrame(0, inputFrameRelHeight, 1, outputFrameRelHeight, backgroundColor1)

    def crtInputFrame(self, relx, rely, relwidth, relheight, bg):
        self.inputFrame = Frame(self.generalFrame, bg=bg)
        self.inputFrame.place(relx=relx, rely=rely, relwidth=relwidth, relheight=relheight)

        column = 0
        for part in self.inputParts:
            self.inputParts[part] = Frame(self.inputFrame, bg=bg)
            self.inputParts[part].grid(row=0, column=column, padx=20)
            column += 1
            header = Label(self.inputParts[part], text=part, font=(fontType, 20, 'bold'), fg=foregroundColor,
                           bg=bg)

            header.grid(row=0, column=0)
            self.crtInputElements(part)

    def crtOutputFrame(self, relx, rely, relwidth, relheight, bg):
        self.outputFrame = Frame(self.generalFrame, bg=bg)
        self.outputFrame.place(relx=relx, rely=rely, relwidth=relwidth, relheight=relheight)

        self.resultLabel = crtLabel(self.outputFrame, {'row': 0, 'column': 0, 'padx': 0, 'pady': 0})

        self.txtArea = crtOutputTextArea(self.outputFrame, {'relx': 0, 'rely': 0, 'relheight': 1, 'relwidth': 1})

    def crtInputElements(self, part):
        if part == 'Data Part':
            self.crtDataPartElements(part)

        elif part == 'Roberta Part':
            self.crtRobertaPartElements(part)

        elif part == 'Prepare Data Part':
            self.crtPrepareDataPartElements(part)

        elif part == 'Predict Part':
            self.crtPredictPartElements(part)

        elif part == 'Test Data Part':
            self.crtTestDataPartElements(part)

    def crtDataPartElements(self, part):
        # crypto
        cryptoBtn = crtMenuBtn(self.inputParts[part], "Crypto", {'row': 1, 'column': 0, 'padx': 0, 'pady': 15},
                               self.cryptoList, self.setCryptoType, "dataScpCrypto")
        self.inputs['dataScpCrypto'][1] = crtLabel(self.inputParts[part],
                                                   {'row': 1, 'column': 1, 'padx': 0, 'pady': 15})

        # start date
        startDateBtn = crtBtn(self.inputParts[part], "Start Date", {'row': 2, 'column': 0, 'padx': 0, 'pady': 15})
        startDateBtn.config(command=partial(self.crtCalender, self.inputParts[part], 'dataScpStartDate'))
        self.inputs['dataScpStartDate'][1] = crtLabel(self.inputParts[part],
                                                      {'row': 2, 'column': 1, 'padx': 0, 'pady': 15})

        # end date
        endDateBtn = crtBtn(self.inputParts[part], "End Date", {'row': 3, 'column': 0, 'padx': 0, 'pady': 15})
        endDateBtn.config(command=partial(self.crtCalender, self.inputParts[part], 'dataScpEndDate'))
        self.inputs['dataScpEndDate'][1] = crtLabel(self.inputParts[part],
                                                    {'row': 3, 'column': 1, 'padx': 0, 'pady': 15})

        # scrape data
        scrapeDataBtn = crtBtn(self.inputParts[part], "Scrape Data", {'row': 4, 'column': 0, 'padx': 0, 'pady': 15})
        scrapeDataBtn.config(command=self.dataScraping)
        self.dataScpLabel = crtLabel(self.inputParts[part], {'row': 4, 'column': 1, 'padx': 0, 'pady': 15})
        self.dataScpLabel.config(fg=finishForegroundColor)

    def crtRobertaPartElements(self, part):
        # Crypto
        cryptoBtn = crtMenuBtn(self.inputParts[part], "Crypto", {'row': 1, 'column': 0, 'padx': 0, 'pady': 15},
                               self.cryptoList, self.setCryptoType, "robertaCrypto")
        self.inputs['robertaCrypto'][1] = crtLabel(self.inputParts[part],
                                                   {'row': 1, 'column': 1, 'padx': 0, 'pady': 15})

        # Show Plot
        self.crtShowPlotBtn(self.inputParts[part], {'row': 2, 'column': 0, 'padx': 0, 'pady': 15}, 'robertaShowPlot')

        # Train Roberta
        trainBtn = crtBtn(self.inputParts[part], "Train Roberta", {'row': 3, 'column': 0, 'padx': 0, 'pady': 15})
        trainBtn.config(command=partial(self.trainRoberta, 'robertaShowPlot'))
        self.robertaTrainLabel = crtLabel(self.inputParts[part], {'row': 3, 'column': 1, 'padx': 0, 'pady': 15})
        self.robertaTrainLabel.config(fg=finishForegroundColor)

    def crtPrepareDataPartElements(self, part):
        # Crypto
        cryptoBtn = crtMenuBtn(self.inputParts[part], "Crypto", {'row': 1, 'column': 0, 'padx': 0, 'pady': 15},
                               self.cryptoList, self.setCryptoType, "dataCrypto")
        self.inputs['dataCrypto'][1] = crtLabel(self.inputParts[part],
                                                {'row': 1, 'column': 1, 'padx': 0, 'pady': 15})

        # Show Plot
        self.crtShowPlotBtn(self.inputParts[part], {'row': 2, 'column': 0, 'padx': 0, 'pady': 15}, 'dataShowPlot')

        # Prepare Roberta
        prepareBtn = crtBtn(self.inputParts[part], "Prepare Data", {'row': 3, 'column': 0, 'padx': 0, 'pady': 15})
        prepareBtn.config(command=self.prepareData)
        self.dataPrepareLabel = crtLabel(self.inputParts[part], {'row': 3, 'column': 1, 'padx': 0, 'pady': 15})
        self.dataPrepareLabel.config(fg=finishForegroundColor)

    def crtPredictPartElements(self, part):
        # Crypto
        cryptoBtn = crtMenuBtn(self.inputParts[part], "Crypto", {'row': 1, 'column': 0, 'padx': 0, 'pady': 15},
                               self.cryptoList, self.setCryptoType, "predictCrypto")
        self.inputs['predictCrypto'][1] = crtLabel(self.inputParts[part],
                                                   {'row': 1, 'column': 1, 'padx': 0, 'pady': 15})

        # Show Plot
        self.crtShowPlotBtn(self.inputParts[part], {'row': 1, 'column': 2, 'padx': 0, 'pady': 15}, 'predictShowPlot')

        # Train Data Start Date
        trainStartDateBtn = crtBtn(self.inputParts[part], "Train Start Date",
                                   {'row': 2, 'column': 0, 'padx': 0, 'pady': 15})
        trainStartDateBtn.config(command=partial(self.crtCalender, self.inputParts[part], 'trainDataStartDate'))
        self.inputs['trainDataStartDate'][1] = crtLabel(self.inputParts[part],
                                                        {'row': 2, 'column': 1, 'padx': 0, 'pady': 15})

        # Train Data End Date
        trainEndDateBtn = crtBtn(self.inputParts[part], "Train End Date",
                                 {'row': 2, 'column': 2, 'padx': 0, 'pady': 15})
        trainEndDateBtn.config(command=partial(self.crtCalender, self.inputParts[part], 'trainDataEndDate'))
        self.inputs['trainDataEndDate'][1] = crtLabel(self.inputParts[part],
                                                      {'row': 2, 'column': 3, 'padx': 0, 'pady': 15})

        # Test Data Start Date
        testStartDateBtn = crtBtn(self.inputParts[part], "Test Start Date",
                                  {'row': 3, 'column': 0, 'padx': 0, 'pady': 15})
        testStartDateBtn.config(command=partial(self.crtCalender, self.inputParts[part], 'testDataStartDate'))
        self.inputs['testDataStartDate'][1] = crtLabel(self.inputParts[part],
                                                       {'row': 3, 'column': 1, 'padx': 0, 'pady': 15})

        # Test Data End Date
        testEndDateBtn = crtBtn(self.inputParts[part], "Test End Date",
                                {'row': 3, 'column': 2, 'padx': 0, 'pady': 15})
        testEndDateBtn.config(command=partial(self.crtCalender, self.inputParts[part], 'testDataEndDate'))
        self.inputs['testDataEndDate'][1] = crtLabel(self.inputParts[part],
                                                     {'row': 3, 'column': 3, 'padx': 0, 'pady': 15})

        # Predict
        predictBtn = crtBtn(self.inputParts[part], "Predict", {'row': 4, 'column': 0, 'padx': 0, 'pady': 15})
        predictBtn.config(command=partial(self.classificationPrediction, 'predictShowPlot'))
        self.predictLabel = crtLabel(self.inputParts[part], {'row': 4, 'column': 1, 'padx': 0, 'pady': 15})
        self.predictLabel.config(fg=finishForegroundColor)

    def crtTestDataPartElements(self, part):
        # Start Index
        starIndexEntry = crtEntry(self.inputParts[part], {'row': 1, 'column': 0, 'padx': 15, 'pady': 0}, "Start Index")
        starIndexBtn = crtBtn(self.inputParts[part], "Select", {'row': 3, 'column': 0, 'padx': 0, 'pady': 15})
        starIndexBtn.config(command=partial(self.getIntegerInput, starIndexEntry, 'startIndex'))
        self.inputs['startIndex'][1] = crtLabel(self.inputParts[part], {'row': 3, 'column': 1, 'padx': 0, 'pady': 15})

        # End Index
        endIndexEntry = crtEntry(self.inputParts[part], {'row': 1, 'column': 2, 'padx': 15, 'pady': 0}, "End Index")
        endIndexBtn = crtBtn(self.inputParts[part], "Select", {'row': 3, 'column': 2, 'padx': 0, 'pady': 15})
        endIndexBtn.config(command=partial(self.getIntegerInput, endIndexEntry, 'endIndex'))
        self.inputs['endIndex'][1] = crtLabel(self.inputParts[part], {'row': 3, 'column': 3, 'padx': 0, 'pady': 15})

        # Test Datas Predict
        testDatasPredictBtn = crtBtn(self.inputParts[part], "Test Datas Predict",
                                     {'row': 4, 'column': 0, 'padx': 0, 'pady': 15})
        testDatasPredictBtn.config(command=self.testDatasPrediction)
        self.testDatasPredictLabel = crtLabel(self.inputParts[part], {'row': 4, 'column': 1, 'padx': 0, 'pady': 15})
        self.testDatasPredictLabel.config(fg=finishForegroundColor)

        # Test Datas Average Of Consistency
        testDatasAvgBtn = crtBtn(self.inputParts[part], "Take Average Of\nConsistency",
                                 {'row': 4, 'column': 2, 'padx': 0, 'pady': 15})
        testDatasAvgBtn.config(command=self.testDatasAvg)
        self.testDatasAvgLabel = crtLabel(self.inputParts[part], {'row': 4, 'column': 3, 'padx': 0, 'pady': 15})
        self.testDatasAvgLabel.config(fg=finishForegroundColor)

    def getDate(self, calender, var):
        selectedDate = calender.get_date()
        self.inputs[var][0] = selectedDate
        self.inputs[var][1].config(text=selectedDate)

    def crtCalender(self, frame, var):
        calenderWindow = Toplevel(frame)
        calenderWindow.title("Calender")
        calenderWindow.geometry()
        calenderWindow.resizable(False, False)
        calender = Calendar(calenderWindow, selectmode="day", date_pattern="yyyy-mm-dd")
        calender.pack()

        btn = Button(calenderWindow, text="Select Date", command=partial(self.getDate, calender, var),
                     fg=btnForegroundColor, bg=btnBackgroundColor,
                     activeforeground=btnActiveForegroundColor, activebackground=btnActiveBackgroundColor,
                     font=(btnFontType, btnFontSize))
        btn.pack(side=BOTTOM, fill=BOTH)

    def setCryptoType(self, crypto, type):
        self.inputs[type][0] = crypto
        self.inputs[type][1].config(text=crypto)

        if type == 'dataScpCrypto':
            if not os.path.exists(f'Plot Files/{crypto}'):
                os.makedirs(f'Plot Files/{crypto}')

            if not os.path.exists(f'Data Files/{crypto}/csv Files'):
                os.makedirs(f'Data Files/{crypto}/csv Files')
                os.makedirs(f'Data Files/{crypto}/xlsx Files')

    def dateCompare(self, startDate, endDate):
        if startDate > endDate:
            return False
        return True

    # Data Scraping Functions

    def dataScraping(self):
        startDate = self.inputs["dataScpStartDate"][0]
        endDate = self.inputs["dataScpEndDate"][0]
        crypto = self.inputs["dataScpCrypto"][0]

        self.dataScpLabel.config(text="")

        if startDate is None or endDate is None or crypto is None or not self.dateCompare(startDate, endDate):
            tkinter.messagebox.showwarning(title="Error", message="Invalid Inputs")
            return

        newData = data.Data(startDate, endDate, crypto)
        ds.dataScraping(newData)

        self.dataScpLabel.config(text="Calculated ✓")

    # Roberta Functions

    def trainRoberta(self, showPlot):
        crypto = self.inputs['robertaCrypto'][0]
        label = self.robertaTrainLabel
        if crypto is not None:
            label.config(text="")
            exec(open('RobertaModel/roberta_training.py').read(),
                 {'symbol': crypto, 'showPlot': self.inputs[showPlot].get()})
            label.config(text="Trained ✓")
        else:
            tkinter.messagebox.showwarning(title="Error", message="Choose Crypto Type")

    # Prepare Data Functions

    def prepareData(self):
        crypto = self.inputs['dataCrypto'][0]
        label = self.dataPrepareLabel
        if crypto is not None:
            label.config(text="")
            exec(open('PrepareDatas/prepare_data.py').read(), {'symbol': crypto})
            label.config(text="Prepared ✓")
        else:
            tkinter.messagebox.showwarning(title="Error", message="Choose Crypto Type")

    # Predict Part Functions

    def classificationPrediction(self, showPlot):
        self.txtArea.config(state='normal')
        self.txtArea.delete('1.0', END)

        crypto = self.inputs['predictCrypto'][0]
        label = self.predictLabel
        trainStartDate = self.inputs['trainDataStartDate'][0]
        trainEndDate = self.inputs['trainDataEndDate'][0]
        testStartDate = self.inputs['testDataStartDate'][0]
        testEndDate = self.inputs['testDataEndDate'][0]

        if crypto is not None or self.dateCompare(trainStartDate, trainEndDate) or \
                self.dateCompare(testStartDate, testEndDate) or self.dateCompare(trainEndDate, testStartDate):

            label.config(text="")

            inf = {
                'Crypto': crypto,
                'Train Start Date': trainStartDate,
                'Train End Date': trainEndDate,
                'Test Start Date': testStartDate,
                'Test End Date': testEndDate
            }

            ta.manuel(inf)

            label.config(text="Predicted ✓")

            movementPred = ta.movementPred

            for date, predicted in movementPred.items():
                result = "It is predicted to rise" if predicted == 1 else "It is predicted to fall"
                date = str(date).split(" ")[0]
                txt = f"{date}: {result}\n"
                self.txtArea.insert('end', txt)

            self.txtArea.config(state='disable')

        else:
            tkinter.messagebox.showwarning(title="Error", message="Invalid Inputs")

    def testDatasPrediction(self):
        startIndex = self.inputs['startIndex'][0]
        endIndex = self.inputs['endIndex'][0]

        if startIndex > endIndex:
            tkinter.messagebox.showwarning(title="Error", message="Start Index Cannot Be Greater Than End Index")
            return

        label = self.testDatasPredictLabel
        label.config(text="")
        ta.automation(startIndex, endIndex)
        label.config(text="Predicted ✓")

    def testDatasAvg(self):
        startIndex = self.inputs['startIndex'][0]
        endIndex = self.inputs['endIndex'][0]

        if startIndex > endIndex:
            tkinter.messagebox.showwarning(title="Error", message="Start Index Cannot Be Greater Than End Index")
            return

        label = self.testDatasAvgLabel
        label.config(text="")
        ta.calculateAvgOfConsistency(startIndex, endIndex, self.txtArea)
        label.config(text="Calculated ✓")

    def crtShowPlotBtn(self, frame, layout, var):
        showPlotBtn = Checkbutton(frame, text="Show Plots", variable=self.inputs[var],
                                  onvalue=True, offvalue=False)
        showPlotBtn.config(fg=btnForegroundColor, bg=btnBackgroundColor,
                           activeforeground=btnActiveForegroundColor,
                           activebackground=btnActiveBackgroundColor)
        showPlotBtn.config(font=(btnFontType, btnFontSize))
        showPlotBtn.grid(row=layout['row'], column=layout['column'], padx=layout['padx'], pady=layout['pady'])

    def getIntegerInput(self, entry, var):
        try:
            value = int(entry.get())
            self.inputs[var][0] = value
            self.inputs[var][1].config(text=value)
        except ValueError:
            tkinter.messagebox.showwarning(title="Error", message="Invalid Input Enter Integer")

    def on_close(self):
        exit()


def crtBtn(frame, text, layout):  # layout include row, column, padx, pady
    btn = Button(frame, text=text)
    btn.config(fg=btnForegroundColor, bg=btnBackgroundColor, activeforeground=btnActiveForegroundColor,
               activebackground=btnActiveBackgroundColor)
    btn.config(font=(btnFontType, btnFontSize))
    btn.grid(row=layout['row'], column=layout['column'], padx=layout['padx'], pady=layout['pady'])
    return btn


def crtMenuBtn(frame, text, layout, list, command, input):  # layout include row, column, padx, pady
    btn = Menubutton(frame, text=text)
    btn.config(fg=btnForegroundColor, bg=btnBackgroundColor, activeforeground=btnActiveForegroundColor,
               activebackground=btnActiveBackgroundColor)
    btn.config(font=(btnFontType, btnFontSize))
    btn.grid(row=layout['row'], column=layout['column'], padx=layout['padx'], pady=layout['pady'])

    menu = Menu(btn, tearoff=0)
    for item in list:
        menu.add_command(label=item, command=partial(command, item, input))

    btn['menu'] = menu

    return btn


def crtLabel(frame, layout):  # layout include row, column, padx, pady
    label = Label(frame, text="")
    label.config(fg=foregroundColor, bg=backgroundColor2)
    label.config(font=(btnFontType, btnFontSize))
    label.grid(row=layout['row'], column=layout['column'], padx=layout['padx'], pady=layout['pady'])
    return label


def crtOutputTextArea(frame, place):
    txtArea = Text(frame, state='disabled')
    txtArea.config(fg=btnForegroundColor, bg=btnBackgroundColor)
    txtArea.config(font=(fontType, fontSize))
    txtArea.place(relx=place['relx'], rely=place['rely'], relheight=place['relheight'], relwidth=place['relwidth'])

    scroll_y = Scrollbar(txtArea)
    scroll_y.pack(side=RIGHT, fill=Y)

    txtArea.config(yscrollcommand=scroll_y.set)
    scroll_y.config(command=txtArea.yview)

    return txtArea


def crtEntry(frame, layout, txt):
    label = crtLabel(frame, layout)
    label.config(text=txt)
    entry = Entry(frame, width=5)
    entry.config(fg=foregroundColor, bg=backgroundColor1)
    entry.config(font=(fontType, fontSize))
    entry.grid(row=layout['row'] + 1, column=layout['column'], padx=layout['padx'], pady=layout['pady'])

    return entry
