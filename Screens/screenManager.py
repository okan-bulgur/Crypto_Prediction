from tkinter import *

from Screens import menubar
from Screens import mainScreen as ms
from Screens import userVersionScreen as uvs


class screenManager:
    window = None
    mainFrame = None

    title = "Crypto Prediction"

    screenWidth = 1500
    screenHeight = 700

    fontSize = 12
    fontType = 'Arial'

    backgroundColor = '#FFFBF5'
    foregroundColor = '#A75D5D'

    btnBackgroundColor = '#D3756B'
    btnForegroundColor = '#F9F5E7'
    hoverBtnBackgroundColor = '#F0997D'
    hoverBtnForegroundColor = '#F9F5E7'

    def __init__(self):
        self.window = Tk()
        self.window.title(self.title)
        icon = PhotoImage(file='res/btcIcon.png')
        self.window.iconphoto(True, icon)
        self.openUserScreen()
        menubar.createMenuBar(self, self.window)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def setTitle(self, title):
        self.title = title

    def openMainScreen(self):
        self.window.geometry(f'{ms.screenWidth}x{ms.screenHeight}')
        if self.mainFrame:
            self.mainFrame.forget()
        self.window.title(ms.title)
        self.mainFrame = Frame(self.window)
        ms.MainScreen(self.mainFrame)
        self.mainFrame.pack(expand=True, fill=BOTH)

    def openUserScreen(self):
        self.window.geometry(f'{uvs.screenWidth}x{uvs.screenHeight}')
        if self.mainFrame:
            self.mainFrame.forget()
        self.window.title(uvs.title)
        self. mainFrame = Frame(self.window)
        uvs.UserScreen(self.mainFrame)
        self.mainFrame.pack(expand=True, fill=BOTH)

    def on_close(self):
        exit()