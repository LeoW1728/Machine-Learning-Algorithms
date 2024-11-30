class Item: # 一条数据/一个物体
    __slots__ = {'properties', 'type'}
    def __init__(self, type : str = 'Item', **properties):
        self.type = type
        if properties:
            self.properties = dict(properties)
        else:
            self.properties = {}
    def __getitem__(self, index): # 此下三个方法用于简化代码，无实际用处
        return self.properties[index]
    def __setitem__(self, index, value):
        self.properties[index] = value
    def __delitem__(self, index):
        del self.properties[index]
    def copy(self, feats): # 复制一条数据/一个物体，用于生成一个子数据集
        newitem = Item(self.type)
        for feat in feats:
            if self[feat]:
                newitem[feat] = self[feat]
        return newitem
    def __iter__(self): # 简化代码
        return iter(self.properties.items())
    def __str__(self) -> str: # 格式化打印
        itemstr = '<{1} object at {0}'.format(
            hex(id(self)), self.type)
        for prop, pvalue in self.properties.items():
            itemstr += ', {0}: {1}'.format(prop, pvalue)
        itemstr += '>'
        return itemstr
def Melon(**melon_properties): # 定义西瓜构造函数，其他类型的数据也可以使用类似的方法
    I = Item('Melon', **melon_properties)
    return I