from math import log2
from Item import Item

class Data:
    feats : list[str] # 所有数据可能拥有的所有属性列表
    items : list[Item] # 数据集中的数据列表
    __slots__ = {'feats', 'items'}
    def __init__(self, feats: list, *items): # 使用 feats 和数据 items 初始化
        self.feats = feats
        if items:
            self.items = list(items)
        else:
            self.items = []
    def append(self, item) -> None: # 添加一条数据
        for property, value in item.properties.items():
            if property not in self.feats:
                self.feats.append(property)
        self.items.append(item)
    def __len__(self) -> int: # 数据集的大小
        return len(self.items)
    def __setitem__(self, index, item): # 简化代码用
        self.items[index] = item
    def __getitem__(self, index):
        return self.items[index]
    def __delitem__(self, index):
        del self.items[index]
    def __iter__(self):
        return iter(self.items)
    def __str__(self): # 格式化打印
        datastr = '<Data object at {0}, data: [\n'.format(hex(id(self)))
        for datum in self.items:
            datastr += '| ' + str(datum) + '\n'
        datastr += ']>'
        return datastr
    def subdata_values(self, feat, *, mode = 'normal'):
        # 根据 mode 和所选属性 feat 生成一个子集的 feat 值的列表
        subdatav = []
        if mode == 'normal':
            for item in self.items:
                if item[feat]:
                    subdatav.append(item[feat])
            return subdatav
        elif mode == 'append':
            for item in self.items:
                if item[feat]:
                    subdatav.append((item[self.main_feat], item[feat]))
            return subdatav
    def count(self, feat : str) -> str: # 统计含有 feat 属性的数据条数
        if feat not in self.feats:
            return 0
        count = 0
        for item in self.items:
            if item[feat]:
                count += 1
        return count
    def subdata(self, feats : list, conditions : dict): # 生成子集
        # 在 conditions 条件限制下保留 feats 生成子集
        # 如 condition = {'quality': 'good'} -> 'quality' 为 'good' 生成一个只有好瓜的子集
        # 如 feats = ['quality'] 时生成的子集里的数据只有 'quality' 一个属性
        subdata = Data(feats)
        if conditions:
            for datum in self.items:
                for condition, cvalue in conditions.items():
                    if datum[condition] != cvalue:
                        break
                else:
                    subdata.append(datum.copy(feats))
        else:
            for datum in self.items:
                subdata.append(datum.copy(feats))
        return subdata
    def feat_val(self, feat) -> set: # 某个属性 feat 所有可能的取值，比如瓜的 'quality' 可取 'good' 和 'bad'
        feat_set = set([item[feat] for item in self.items])
        return feat_set
    def feat_dict(self, feat, * , mode = 'normal') -> dict: # 将某个属性 feat 的取值与一个数集对应，返回对应关系字典
        feat_set = set([item[feat] for item in self.items])
        for i in feat_set:
            if not isinstance(i, (int, float, complex)):
                break
        else:
            return dict()
        if mode == 'normal': # normal 返回的对应数集是 [0,1,2,...]
            feat_d = dict(zip(feat_set, range(len(feat_set))))
        elif mode == 'compile': # compile 返回的对应数集是 [0, 1 / l, 2 / l, ..., (l - 1) / l]，其中 l 是 feat 的取值可能总数
            l = len(feat_set)
            m = [i/l for i in range(l)]
            feat_d = dict(zip(feat_set, m))
        return feat_d