class Item:
    def __init__(self, **properties):
        if properties:
            self.properties = dict(properties)
        else:
            self.properties = {}

    def __getitem__(self, index):
        return self.properties[index]

    def __setitem__(self, index, value):
        self.properties[index] = value

    def __delitem__(self, index):
        del self.properties[index]

    def copy(self, feats):
        newitem = Item()
        for feat in feats:
            if self[feat]:
                newitem[feat] = self[feat]
        return newitem
    
    def __iter__(self):
        return iter(self.properties.items())
  
class Melon(Item):
    def __init__(self, **properties):
        super().__init__(**properties)

    def __str__(self) -> str:
        melonstr = '<Melon object at {0}'.format(
            str(hex(id(self)))
        )
        for prop, pvalue in self.properties.items():
            melonstr += ', {0}: {1}'.format(prop, pvalue)

        melonstr += '>'
        return melonstr

    # def __getitem__(self, index):
    #     return self.properties[index]

    # def __setitem__(self, index, value):
    #     self.properties[index] = value

    # def __delitem__(self, index):
    #     del self.properties[index]
    
    def copy(self, feats):
        newmelon = Melon()
        for property in self.properties:
            if property in feats:
                newmelon[property] = self[property]
        return newmelon
