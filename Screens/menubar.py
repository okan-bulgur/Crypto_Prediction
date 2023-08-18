from tkinter import Menu


def createMenuBar(sm, window):
    menubar = Menu()
    window.config(menu=menubar)

    fileMenu = Menu(menubar, tearoff=0)
    menubar.add_command(label="Main Screen", command=lambda: sm.openMainScreen())
    menubar.add_command(label="User Screen", command=lambda: sm.openUserScreen())
