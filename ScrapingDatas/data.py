class Data:
    startDate = None
    endDate = None
    cryptoType = None

    def __init__(self, startDate, endDate, cryptoType):
        self.startDate = startDate
        self.endDate = endDate
        self.cryptoType = cryptoType

    def getStartDate(self):
        return self.startDate

    def getEndDate(self):
        return self.endDate

    def getCryptoType(self):
        return self.cryptoType
