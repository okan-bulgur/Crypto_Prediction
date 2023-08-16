from Screens import mainScreen as ms
from Screens import userVersionScreen as uvs

type = int(input("Enter screen Tyepe\n1)Main Screen\n2)User Screen\n"))
#type = 2

if type == 1:
    ms.MainScreen()
elif type == 2:
    uvs.UserScreen()
else:
    exit()
