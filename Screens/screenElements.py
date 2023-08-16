from functools import partial
from tkinter import *

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
