from math import log2

class Item: # 一条数据/一个物体
    __slots__ = {'properties'}

    def __init__(self, **properties):
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
        newitem = Item()
        for feat in feats:
            if self[feat]:
                newitem[feat] = self[feat]
        return newitem
    
    def __iter__(self):
        return iter(self.properties.items())
  
class Melon(Item): # 定义西瓜为一种物体，这样做可以优化数据可视化结果
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
    
    def copy(self, feats): # 复制一个西瓜
        newmelon = Melon()
        for property in self.properties:
            if property in feats:
                newmelon[property] = self[property]
        return newmelon

class Data: # 数据集，用来（批量）处理数据
    main_feat : str # 用于输出的属性值，即待预测的属性，对于 Melons 为 'quality'
    main_feat_v : list # 待预测属性所有可能的取值，对于 Melons 为 'good' 和 'bad'
    __H : float # 按输出属性值计算得到的熵
    feats : list[str] # 所有物品/数据可能拥有的所有属性列表
    items : list # 数据集中的物品/数据列表

    __slots__ = {'main_feat',
                 'main_feat_v',
                 '__H',
                 'feats',
                 'items'}

    def feat_val(self, feat): # 某个属性 feat 所有可能的取值，比如瓜的 'quality' 可取 'good' 和 'bad'
        feat_set = set([item[feat] for item in self.items]) # different values for items' feature
        return feat_set

    def main_feat_val(self): # 直接设置 main_feat_v
        self.main_feat_v = list(set([item[self.main_feat] for item in self.items])) # different values for items' main feature

    def entropy(self): # 计算熵（根据输出属性，比如瓜的质量：是否为好瓜）
        subdatav = self.subdata_values(self.main_feat)
        subdatav_set = set(subdatav)
        probs, l = [], len(subdatav)
        for p in subdatav_set:
            probs.append(subdatav.count(p) / l)
        H = sum(-prob * log2(prob) for prob in probs if prob != 0)
        return H

    def update(self): # 更新数据集的核心数据
        self.main_feat_val()
        self.__H = self.entropy()

    def __init__(self, main_feat : str, feats: list, *items):
        if main_feat not in feats:
            raise ValueError("Main feat must be included in feats.")
        self.main_feat = main_feat
        self.feats = feats
        if items:
            self.items = list(items)
        else:
            self.items = []

    def append(self, item): # 添加一条数据
        for property, value in item.properties.items():
            if property not in self.feats:
                self.feats.append(property)
        self.items.append(item)

    def __len__(self): # 数据集的大小
        return len(self.items)
    
    def __setitem__(self, index, item): # 简化代码用
        self.items[index] = item
    
    def __getitem__(self, index):
        return self.items[index]

    def __delitem__(self, index):
        del self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __str__(self): # 优化数据集的可视化结果
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

    def count(self, feat : str) -> str: # 统计数据集中含有 feat 属性/标签的数据条数
        if feat not in self.feats:
            return 0
        count = 0
        for item in self.items:
            if item[feat]:
                count += 1
        return count

    def most(self, feat : str) -> str:
        # 给出数据集中在 feat 属性下数据取值最多的一个 feat 的值
        # 比如瓜在 'quality' 下取值最多的是 'good'（有9个好瓜）>'bad'（有8个坏瓜）
        # 如果数量相同则返回优先出现的属性
        if not feat:
            feat = self.main_feat
            most_feat_value, most_count, count = '', 0, 0
            for main_feat_value in self.main_feat_v:
                for item in self.items:
                    if item[feat] == main_feat_value:
                        count += 1
                if count >= most_count:
                    most_feat_value = main_feat_value
                    most_count = count
            return most_feat_value
        else:
            most_feat_value, most_count, count= '', 0, 0
            for feat_value in self.feat_val(feat):
                for item in self.items:
                    if item[feat] == feat_value:
                        count += 1
                if count >= most_count:
                    most_feat_value = feat_value
                    most_count = count
            return most_feat_value

    def subdata(self, feats : list, conditions : dict): # 生成子集
        # 在 conditions 条件限制下保留 feats 生成子集
        # 如 condition = {'quality': 'good'} -> 'quality' 为 'good' 生成一个只有好瓜的子集
        # 如 feats = ['quality'] 时生成的子集里的数据只有 'quality' 一个属性
        subdata = Data(self.main_feat, feats)
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
        subdata.update()
        return subdata

    def gain(self, feat) -> float: # 根据属性 feat 计算信息增益
        if feat == self.main_feat:
            return 0.
        else:
            subdatav = self.subdata_values(feat, mode = 'append')
            if not subdatav:
                return None
            feat_set = self.feat_val(feat)
            D, feat_entropies = len(self), []
            for feat_value in feat_set: # for each possible feature's value calculate different entropy
                feat_subdata = [value[0] for value in subdatav if value[1]==feat_value]
                D_feat = len(feat_subdata)
                main_probs = [feat_subdata.count(main_feat_value)/D_feat for main_feat_value in self.main_feat_v]
                H = -sum(p * log2(p) for p in main_probs if p != 0)
                feat_entropies.append((H,D_feat/D))
            G = -sum(H*P for H, P in feat_entropies)
            return G + self.__H

    def gain_decision(self, feats : list):
        # 根据信息增益的大小做出决策
        # 返回一个字典 {'decision': 在决策树下最佳划分使用的属性,
        #              'pure_gain':列表，列出了每个属性与该属性的信息增益}
        F = feats
        if not feats:
            F = self.feats
        gains = {feat:self.gain(feat) for feat in F if feat in self.feats}
        gains_sorted = sorted(gains.items(), key=lambda x:x[1], reverse=True)
        res = {'decision': gains_sorted[0][0], 'pure_gain':gains_sorted}
        return res

    def IV(self, feat): # intrinsic value 固有值
        if feat == self.main_feat:
            return self.__H
        subdatav = self.subdata_values(feat)
        if not subdatav:
            return None
        subdatav_set = set(subdatav)
        probs, l = [], len(subdatav)
        for p in subdatav_set:
            probs.append(subdatav.count(p) / l)
        iv = sum(-prob * log2(prob) for prob in probs if prob != 0)
        return iv

    def gain_ratio(self, feat): # gain ratio 根据属性 feat 计算该属性的信息增益率
        if feat == self.main_feat:
            return 0
        else:
            subdatav = self.subdata_values(feat, mode = 'append')
            if not subdatav:
                return None
            feat_set = self.feat_val(feat)
            D, feat_entropies = len(self), []
            for feat_value in feat_set: # for each possible feature's value calculate different entropy
                feat_subdata = [value[0] for value in subdatav if value[1]==feat_value]
                D_feat = len(feat_subdata)
                main_probs = [feat_subdata.count(main_feat_value)/D_feat for main_feat_value in self.main_feat_val]
                H = -sum(p * log2(p) for p in main_probs if p != 0)
                feat_entropies.append((H,D_feat/D))
            G = -sum(H*P for H, P in feat_entropies)
            iv = -sum(P * log2(P) for H, P in feat_entropies if P != 0)
            return (G + self.__H) / iv

def register_maker(init_ID): # 注册器（内包），用于注册节点
    now_ID = init_ID
    object_list = {}
    def register(object):
        nonlocal now_ID, object_list
        object_list[now_ID] = object
        now_ID += 1
        return (now_ID - 1)
    return register, object_list

node_register, node_full_list = register_maker(-1)

class Node: # 节点，即决策树的一个分枝处的点
    __feat : any
    __id : int
    kids_ids : list[int]
    __slots__ = {'__feat',      # 本节点的划分属性和选项等信息
                 'kids_ids',    # 本节点所有的子节点的 id 列表
                 '__id'}        # 本节点的 id

    def __str__(self):
        if self.__id == -1:
            return "<Node object: NULL at {0}>".format(hex(id(self)))
        if self.kids_ids:
            node_str = "<Node object #{0}# at {1}, id = {2}, kids_ids = {3}>".format(
                self.__feat, hex(id(self)), self.__id, self.kids_ids)
        else:
            node_str = "<Node object #{0}# at {1}, id = {2}>".format(
                self.__feat, hex(id(self)), self.__id)
        return node_str

    def __init__(self, *, feat = []): # 使用注册器生成新节点
        self.__feat = feat # feat
        self.kids_ids = []
        self.__id = node_register(self)

    def set_feat(self, feat): # 设置节点信息
        self.__feat = feat

    def feat(self): # 返回节点信息
        return self.__feat

    def append(self, kid_id): # 新增子节点（使用 id）
        if kid_id not in self.kids_ids:
            self.kids_ids.append(kid_id)

    def ID(self): # 返回自身的 id
        return self.__id

    def __setitem__(self, index, value): # 简化代码
        self.kids_ids[index] = value

    def __delitem__(self, index):
        del self.kids_ids[index]

    def __getitem__(self, index):
        return self.kids_ids[index]

    def __iter__(self):
        return iter(self.kids_ids)

    def kids(self) -> list: # 返回子节点的 id 列表
        return self.kids_ids
    
Null = Node() # 定义一个空节点

class Tree: # 树结构，用于对节点进行处理
    __slots__ = {'nodes', # 一个字典，存储所有的节点信息
                 'root'}  # 根节点的 id

    def __init__(self, Root : Node): # 使用一个节点 Root 作为根节点生成一个树
        self.nodes = {}
        self.root = Root
        self.nodes[Root.ID()] = Root
    
    def __str__(self): # 优化数据可视化效果
        tree_str = "<Tree object at {0}, nodes are ".format(hex(id(self))) + '{'
        for i, n in self.nodes.items():
            tree_str += '\n' + repr(str(i))+": "+ str(n) +","
        tree_str += "\b}\n>"
        return tree_str

    def __delitem__(self, ID): # 简化代码
        del self.nodes[ID]

    def __getitem__(self, ID):
        return self.nodes[ID]

    def __setitem__(self, ID, node):
        self.nodes[ID] = node

    def __iter__(self):
        return iter(self.nodes.items())

    def append(self, parent : Node, *appending_nodes):
        # 使用 self.append(parent, n1, n2, n3, ...)
        # 会在 parent 节点下添加子节点 n1, n2, n3
        for node in appending_nodes:
            i = node.ID()
            parent.append(i)
            self[i] = node
    
    def parent_node(self, kid : Node):
        # 寻找 kid 节点的 parent（母/父）节点
        if kid.ID() not in self.nodes:
            return Null
        i = kid.ID()
        for t, node in iter(self):
            if i in node.kids():
                i = node.ID()
                break
        else:
            return Null
        return self[i]
            
    def kids(self, node : Node): # 简化代码，kids 返回类似一个列表
        if not node.kids_ids:
            return iter([self[kid_id] for kid_id in node.kids_ids])
        else:
            return None

    def findroot(self): # 返回本树的根节点
        return self.root

    def sort(self): # 使用节点 id 对树的节点列表排序
        self.nodes = dict(sorted(self.nodes.items(), key=lambda x:x[0]))

class DecisionTree: # 决策树模型
    data : Data # 数据集
    feats : list # 属性列表
    tree : Tree # 树结构
    __slots__ = {'data',
                 'feats',
                 'tree'}

    def __init__(self, feats : list, Data : Data): # 使用一个属性列表和数据集生成模型
        self.data = Data
        F = Data.feats
        for feat in feats:
            if feat not in F:
                self.feats = F
                break
        else:
            self.feats = feats
        self.tree = None

    def update_tree(self): # 修正节点错误表达和信息缺失
        for u, v in self.tree.nodes.items():
            if isinstance(v.feat(),str):
                v.set_feat({'node feature':v.feat()})

    def generate_tree(self, data : Data, feats : list) -> Node: # 根据信息增益的原则生成决策树
        if not data: # 若无数据集则不生成树
            return None
        node = Node()
        if not self.tree: # 如果还没有树结构，则生成一个树结构
            self.tree = Tree(node)
        # 根据目前训练使用的属性集提炼出子集
        data = data.subdata(feats,{})
        # 计算目前训练集中样本最多的类
        A = data.most('')
        # 将 输出属性 剔出，防止被用于节点生成分类
        reduced_feats = [feat for feat in data.feats if feat != data.main_feat]
        # 若只剩下一个类
        if len(reduced_feats) == 1:
            node.set_feat(A)
            return node
        # 若无属性可划分或全部数据都属于一类
        Total_H = sum(data.IV(feat) for feat in data.feats)
        if Total_H == 0 or not reduced_feats:
            node.set_feat(A)
            return node
        # 根据信息增益原则选择划分属性
        gd = data.gain_decision(feats)
        chosen_feat, gain = gd['decision'], gd['pure_gain'][0][1]
        if chosen_feat != data.main_feat and gain > 0:
            node.set_feat(chosen_feat)
        else:
            node.set_feat(A)
            return node
        # 根据选择的属性开始生成子节点
        chosen_feat_set = self.data.feat_val(chosen_feat)
        partial_feats = [feat for feat in feats if feat != chosen_feat]
        for feat_value in chosen_feat_set:
            partial_data = data.subdata(partial_feats, {chosen_feat: feat_value})
            if not partial_data:
                node_tag = {'choise':feat_value,'node feature':A}
                kid = Node(feat = node_tag)
            else:
                kid = self.generate_tree(partial_data, partial_feats)
                if isinstance(kid.feat(),str):
                    kid.set_feat({'choise':feat_value,'node feature':kid.feat()})
            self.tree.append(node, kid)
        return node

    def initialize(self): # 根据数据集生成决策树，并直接优化
        self.generate_tree(self.data, self.feats)
        self.update_tree()
        self.tree.sort()

    def predict(self, item: Item): # 对一条数据进行预测分类，即分类任务，需要预先生成决策树
        if self.tree:
            regression, now_id = '', self.tree.root.ID()
            while regression not in self.data.main_feat_v:
                now_node = self.tree[now_id]
                feat = now_node.feat()['node feature']
                if feat in self.data.main_feat_v:
                    regression = feat
                    break
                i = item[feat]
                if now_node.kids_ids:
                    for k in now_node.kids_ids:
                        kid = self.tree[k]
                        c = kid.feat()['choise']
                        if c == i:
                            now_id = kid.ID()
            return regression

# 训练使用的西瓜的数据集
Melons_data = Data('quality', # 瓜的品质是待预测/输出的属性
['tag', 'color', 'root', 'sound', 'texture', 'belly', 'touch' , 'quality'], # 所有属性列表
# 定义每一个瓜的数据
Melon(tag=1, color='green' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=2, color='black' , root='shrink', sound='deep' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=3, color='black' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=4, color='green' , root='shrink', sound='deep' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=5, color='white' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=6, color='green' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='slime' , quality='good'),
Melon(tag=7, color='black' , root='curved', sound='blur' , texture='blur'  , belly='wedged' , touch='slime' , quality='good'),
Melon(tag=8, color='black' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='smooth', quality='good'),
Melon(tag=9, color='black' , root='curved', sound='deep' , texture='blur'  , belly='wedged' , touch='smooth', quality='bad' ),
Melon(tag=10,color='green' , root='erect' , sound='chip' , texture='clear' , belly='smooth' , touch='slime' , quality='bad' ),
Melon(tag=11,color='white' , root='erect' , sound='chip' , texture='bad'   , belly='smooth' , touch='smooth', quality='bad' ),
Melon(tag=12,color='white' , root='shrink', sound='blur' , texture='bad'   , belly='smooth' , touch='slime' , quality='bad' ),
Melon(tag=13,color='green' , root='curved', sound='blur' , texture='blur'  , belly='concave', touch='smooth', quality='bad' ),
Melon(tag=14,color='white' , root='curved', sound='deep' , texture='blur'  , belly='concave', touch='smooth', quality='bad' ),
Melon(tag=15,color='black' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='slime' , quality='bad' ),
Melon(tag=16,color='white' , root='shrink', sound='blur' , texture='bad'   , belly='smooth' , touch='smooth', quality='bad' ),
Melon(tag=17,color='green' , root='shrink', sound='deep' , texture='blur'  , belly='wedged' , touch='smooth', quality='bad' )
)

# 根据这个数据集生成决策树
Melons_data.update() # 优化数据集的表达
###### 不使用 'tag' 进行训练
Melon_Model = DecisionTree([feat for feat in Melons_data.feats if feat != 'tag'], Melons_data)
###### 生成树
Melon_Model.initialize()
###### 打印树的结构
print(Melon_Model.tree)
###### 使用树预测一个西瓜是否为好瓜
Melon18 = Melon(tag=18,color='black' , root='shrink', sound='deep' , texture='blur'  , belly='wedged' , touch='slime')
print(Melon_Model.predict(Melon18))

# 另一个数据集，只有一条数据，仍然可以用于生成决策树
Single_melon_data = Data('quality',
     ['tag', 'color', 'root'  , 'sound', 'texture', 'belly'  , 'touch' , 'quality'],
    Melon(tag=1, color='green' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'))
Single_melon_data.update()
Simple_model = DecisionTree(Single_melon_data.feats, Single_melon_data)
Simple_model.initialize()
###### 由于数据过少，认为该树无论怎么分都是好瓜，即欠拟合
print(Simple_model.tree)