from Neuron import Layer, layers, spawn_layer, neurons, sigmoid, sgn, ReLU, activators, Tanh, Gaussian_kernel
from Data import Data
from Item import Item, Melon

class Network: # 多层神经网络，结构为 输入层 - 隐藏层 - 输出层
    # 当前可以使用的层类型有
    # 'BP' 为 Back Propagation 网络层结构
    #     使用任意的激活函数，每个神经元除权重列表外还有阈值
    # 'RBF' 为 Radial Basis Function 网络层结构
    #     使用线性欧式距离表示径距，使用高斯核作为径向基，
    #     每个神经元除权重列表外还有变量中心列表和核宽度
    # 'RNN' 为 循环神经网络层结构
    #     使用任意的激活函数
    data : Data
    Layer_ids : list[int] # 层顺序
    input_feats : list
    interpreter : dict
    output_feats : list
    learning_rate : float
    activators : dict # 储存各层使用的激活函数信息，根据各层类型调整
    gradients : dict # 储存梯度信息，用于递归计算
    __slots__ = {'data', 'interpreter', 'input_feats', 'output_feats',
                 'learning_rate', 'activators', 'Layer_ids', 'gradients'}
    def set_input_layer(self, input_dimension : int) -> Layer: # 设置输入层
        if input_dimension > 0:
            i = spawn_layer(input_dimension, 'IN')
            i.init()
            return i
    def add_layer(self, layer_dimension : int, prev_layer : Layer,
                         l_t : str = 'BP') -> Layer: # 设置隐藏层
        if layer_dimension > 0 and prev_layer:
            l = spawn_layer(layer_dimension, l_t)
            l.link_prev(prev_layer)
            return l
    def __init__(self, learning_rate = 0.02, # 学习率
                 input_dimension = 1, # 输入维度（必填）
                 *layers : tuple or int) -> None:
        # layers 使用结构 (int : 维数, str : 层类型,
        #       int : 层常用激活函数 ID, *args : 其余参数)
        # [维数无默认，类型默认为 BP, 激活函数默认为 1: sigmoid]
        # 例如使用 Network(.02, 3, 5, (6, 'RBF', 2), (4, 'RBF'))
        # 这将生成一个神经网络，结构为 
        #     输入层(3个神经元) -> 隐藏层(BP, 5个神经元)
        #         -> 隐藏层(RBF，6个神经元, ReLU) - 输出层(RBF，4个神经元)
        self.interpreter = {}
        self.activators = {}
        self.learning_rate = learning_rate
        p = self.set_input_layer(input_dimension)
        self.Layer_ids = [p.ID()]
        if layers:
            for l_info in layers:
                if isinstance(l_info, tuple):
                    l_dim, l_t, *l_args = l_info
                    p = self.add_layer(l_dim, p, l_t)
                    if l_args:
                        self.activators[p.ID()] = l_args[0] if len(
                            l_args) == 1 else l_args
                    else:
                        self.activators[p.ID()] = 1
                elif isinstance(l_info, int):
                    p = self.add_layer(l_info, p)
                    self.activators[p.ID()] = 1
                self.Layer_ids.append(p.ID())
        self.gradients = {l:{} for l in self.Layer_ids}
    def __str__(self) -> str: # 网络的格式化打印
        layers_str = ''
        for l_i in self.Layer_ids:
            layers_str += '\n  ' + str(layers[l_i]) + (', using activator {0},'.format(
                self.activators[l_i]) if l_i in self.activators.keys() else ',')
        layers_str += '\b'
        bp_n_str = '<Neural Network at {0}, Layers = [{1}]>'.format(
            hex(id(self)), layers_str)
        if self.interpreter:
            bp_n_str += '\nInterprets: ' + str(self.interpreter)
        return bp_n_str
    def init(self, compile_mode = 'compile') -> None: # 简化的初始化函数
        D = self.Layer_ids
        d = len(D) - 1
        if self.data:
            for feat in self.data.feats:
                f_d = self.data.feat_dict(feat, mode = compile_mode)
                if f_d:
                    self.interpreter[feat] = f_d
            if self.input_feats:
                self.interpreter['__in'] = dict(zip(self.input_feats,
                                                    layers[D[0]].neuron_dict.keys()))
            if self.output_feats:
                self.interpreter['__out'] = dict(zip(self.output_feats,
                                                     layers[D[d]].neuron_dict.keys()))
        self()
    def __call__(self, mode : str = 'normal') -> None: # 按层顺序激活所有层（除输入层）
        for l_i in self.Layer_ids[1:]:
            f_p = self.activators[l_i]
            layers[l_i](mode, f_p)
    def gradient(self, t : Item) -> float: # 根据一项数据更新梯度值
        E = 0
        D, I = self.Layer_ids, self.interpreter
        d = len(D) - 1
        l, l_d = layers[D[d]], self.gradients[D[d]]
        l_m = l.items()
        for feat in self.output_feats:
            n = I['__out'][feat]
            p = l[n].out
            if feat in I.keys():
                y = I[feat][t[feat]]
            else:
                y = t[feat]
            E += (y - p) ** 2
            l_d[n] = y - p
        d -= 1
        while d > 0:
            # p representing post layer, l representing layer(now)
            # d = dictionary(gradients), m = items, i = id, n = neuron
            p, l = l, layers[D[d]]
            p_d, l_d = l_d, self.gradients[D[d]]
            p_m, l_m = l_m, l.items()
            f_p = activators[self.activators[l.ID()]]
            if p.type() == 'BP':
                l_d['rectified'] = {}
                for l_i, l_n in l_m:
                    if f_p == sigmoid:
                        l_d[l_i] = sum(p_n.inputs[l_i] * p_d[p_i] * p_n.out * (
                                    1 - p_n.out) for p_i, p_n in p_m)
                    else:
                        l_d[l_i] = sum(p_n.inputs[l_i] * p_d[p_i] * f_p(p_n('derive'),
                                    Prime = True) for p_i, p_n in p_m)
            elif p.type() == 'RBF':
                for l_i, l_n in l_m:
                    l_d[l_i] = sum(p_n.inputs[l_i] * p_d[p_i] * Gaussian_kernel(l_n.out - p_n.centers[l_i],
                                p_n.kernel_width, Prime = True) for p_i, p_n in p_m)
            elif p.type() == 'RNN':
                for l_i, l_n in l_m:
                    l_d[l_i] = sum(p_n.inputs[l_i] * p_d[p_i] * f_p(p_n('derive'),
                                    Prime = True) for p_i, p_n in p_m)
            d -= 1
        return E / 2
    def read_data(self, t : Item) -> None: # 读数据，更新各层输出值
        l, I = layers[self.Layer_ids[0]], self.interpreter
        for feat in self.input_feats:
            l_i = I['__in'][feat]
            if feat in I.keys():
                l[l_i].read(I[feat][t[feat]])
            else:
                l[l_i].read(t[feat])
        self()
    def update(self) -> None: # 根据层类型和梯度更新各属性值
        D, r = self.Layer_ids, self.learning_rate
        d = len(D) - 1
        l, l_d = layers[D[d]], self.gradients[D[d]]
        l_m = l.items()
        d -= 1
        while d >= 0:
            p, l = l, layers[D[d]]
            p_d, l_d = l_d, self.gradients[D[d]]
            p_m, l_m = l_m, l.items()
            f_p = activators[self.activators[p.ID()]]
            if p.type() == 'BP':
                for p_i, p_n in p_m:
                    v = f_p(p_n(mode = 'derive'), Prime = True)
                    for l_i, l_n in l_m:
                        p_n.inputs[l_i] += r * l_n.out * p_d[p_i] * v
                        p_n.threshold -= r * p_d[p_i] * v
            elif p.type() == 'RBF':
                for p_i, p_n in p_m:
                    for l_i, l_n in l_m:
                        p_n.inputs[l_i] += r * p_d[p_i] * Gaussian_kernel(
                            l_n.out - p_n.centers[l_i], p_n.kernel_width)
                        p_n.centers[l_i] -= r * p_d[p_i] * Gaussian_kernel(
                            l_n.out - p_n.centers[l_i], p_n.kernel_width,
                            Prime = True) * p_n.inputs[l_i]
                        # 若使用此段代码，则将更新神经元核宽度
                        # qc = r * p_d[p_i] * Gaussian_kernel(
                        #     l_n.out - p_n.centers[l_i], p_n.kernel_width)
                        # qs = p_n.inputs[l_i] * (l_n.out - p_n.centers[l_i]) ** 2 / p_n.kernel_width ** 3
                        # p_n.kernel_width += qc * qs
            elif p.type() == 'RNN':
                for p_i, p_n in p_m:
                    v = f_p(p_n(mode = 'derive'), Prime = True)
                    for l_i, l_n in l_m:
                        p_n.inputs[l_i] += r * l_n.out * p_d[p_i] * v
                    for pp_i, pp_n in p_m:
                        p_n.memory_inputs[pp_i] += r * pp_n.memory * p_d[p_i] * v
                for p_i, p_n in p_m:
                    p_n.memorize()
            d -= 1
    def train_data(self, item_id : int) -> float: # 读取一条数据并据此更新网络的权重
        # 读取一条数据
        t = self.data[item_id]
        self.read_data(t)
        E = self.gradient(t) # E 返回本项数据的方均误差，gradient 方法同时更新梯度
        self.update() # 根据梯度更新各层连接权、阈值、中心值等
        return E
    def train(self, train_list : list = [][:]) -> list:
        # 根据数据的 ID 列表 train_list 训练，若列表为空则根据整个数据集训练
        # 返回一个列表，第一项为累积误差，第二项为方均误差列表，第三项为最大的方均误差，最后为系统最大绝对误差
        E = []
        l = train_list if train_list else range(len(self.data))
        for datum in l:
            e = self.train_data(datum)
            # print(self.output_layer)
            E.append(e)
            Q = sum(E) / len(E)
        return [Q, E, max(E), (2 * max(E)) ** .5]
    def predict(self, item) -> dict: # 根据一条数据计算输出
        t, D = item, self.Layer_ids
        i = layers[D[0]]
        for feat in self.input_feats:
            n = self.interpreter['__in'][feat]
            try:
                if feat in self.interpreter.keys():
                    i[n].read(self.interpreter[feat][t[feat]])
                else:
                    i[n].read(t[feat])
            except KeyError:
                print("Something (feat : %s) missing in datum needed to be predicted." % feat)
        self()
        d = {feat: layers[D[len(D) - 1]][self.interpreter['__out'][feat]].out for feat in self.output_feats}
        return d
    def test(self, item_id): # 将数据集中的一条数据与输出比较测试
        p = self.predict(self.data.items[item_id])
        y = {}
        for feat in self.output_feats:
            t = self.data.items[item_id][feat]
            if feat in self.interpreter.keys():
                y[feat] = self.interpreter[feat][t]
            else:
                y[feat] = t
        print('\nTesting:   ', item_id)
        print('Predicted:', p)
        print('Observed: ', y,'\n')
    def test_list(self, item_id_list = [][:]):
        # 测试一个数据集，若 item_id_list 为空则测试整个数据集
        i = item_id_list if item_id_list else range(len(self.data))
        for datum in i:
            self.test(datum)
    def load_data(self, data: Data,
                  input_feats : list = [][:],
                  output_feats : list = [][:]) -> None: # 加载数据集
        self.data = data
        D = self.Layer_ids
        d = len(D) - 1
        if input_feats:
            if len(input_feats) == len(layers[D[0]]): # 设置输入属性，必须和输入层对齐
                self.input_feats = input_feats
        if output_feats:
            if len(output_feats) == len(layers[D[d]]): # 设置输出属性，必须和输出层对齐
                self.output_feats = output_feats
    def set_up_feats(self,
                     input_feats : list = [][:],
                     output_feats : list = [][:]) -> None: # 设置输入/出属性，必须和输入/出层对齐
        D = self.Layer_ids
        if input_feats:
            if len(input_feats) == len(layers[D[0]]):
                self.input_feats = input_feats
        if output_feats:
            if len(output_feats) == len(layers[D[d]]):
                self.output_feats = output_feats
##################################################### 一个例子
# 设置神经网络基础结构
Melon_Network = Network(.05, 8, # 学习率，输入层
                        5, 1) # 其他层
# 训练使用的西瓜的数据集
Melons_data = Data(
['tag', 'color', 'root', 'sound', 'texture', 'belly', 'touch' , 'density', 'sugar', 'quality'], # 所有属性列表
# 定义每一个瓜的数据
Melon(tag=1, color='green' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', density = 0.697, sugar = 0.460, quality='good'),
Melon(tag=2, color='black' , root='shrink', sound='deep' , texture='clear' , belly='concave', touch='smooth', density = 0.774, sugar = 0.376, quality='good'),
Melon(tag=3, color='black' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', density = 0.634, sugar = 0.264, quality='good'),
Melon(tag=4, color='green' , root='shrink', sound='deep' , texture='clear' , belly='concave', touch='smooth', density = 0.608, sugar = 0.318, quality='good'),
Melon(tag=5, color='white' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', density = 0.556, sugar = 0.215, quality='good'),
Melon(tag=6, color='green' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='slime' , density = 0.403, sugar = 0.237, quality='good'),
Melon(tag=7, color='black' , root='curved', sound='blur' , texture='blur'  , belly='wedged' , touch='slime' , density = 0.481, sugar = 0.149, quality='good'),
Melon(tag=8, color='black' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='smooth', density = 0.437, sugar = 0.211, quality='good'),
Melon(tag=9, color='black' , root='curved', sound='deep' , texture='blur'  , belly='wedged' , touch='smooth', density = 0.666, sugar = 0.091, quality='bad' ),
Melon(tag=10,color='green' , root='erect' , sound='chip' , texture='clear' , belly='smooth' , touch='slime' , density = 0.243, sugar = 0.267, quality='bad' ),
Melon(tag=11,color='white' , root='erect' , sound='chip' , texture='bad'   , belly='smooth' , touch='smooth', density = 0.245, sugar = 0.057, quality='bad' ),
Melon(tag=12,color='white' , root='shrink', sound='blur' , texture='bad'   , belly='smooth' , touch='slime' , density = 0.343, sugar = 0.099, quality='bad' ),
Melon(tag=13,color='green' , root='curved', sound='blur' , texture='blur'  , belly='concave', touch='smooth', density = 0.639, sugar = 0.161, quality='bad' ),
Melon(tag=14,color='white' , root='curved', sound='deep' , texture='blur'  , belly='concave', touch='smooth', density = 0.657, sugar = 0.198, quality='bad' ),
Melon(tag=15,color='black' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='slime' , density = 0.360, sugar = 0.370, quality='bad' ),
Melon(tag=16,color='white' , root='shrink', sound='blur' , texture='bad'   , belly='smooth' , touch='smooth', density = 0.593, sugar = 0.042, quality='bad' ),
Melon(tag=17,color='green' , root='shrink', sound='deep' , texture='blur'  , belly='wedged' , touch='smooth', density = 0.719, sugar = 0.103, quality='bad' )
)
# 神经网络加载数据集
Melon_Network.load_data(Melons_data,
    ['color', 'root', 'sound', 'texture', 'belly', 'touch', 'density', 'sugar'], # 待输入的属性，对齐 6 个输入神经元
    [ 'quality']) # 待预测的属性，对齐 3 个输出神经元
Melon_Network.init('normal') # 初始化
# print(Melon_Network)
M = 3000 # 训练总次数
new_err = 0
for i in range(100): # 训练 1 次更新 err 项
    err = new_err
    d = Melon_Network.train()
    new_err = d[0]
for i in range(M): # 训练，并显示每次训练的系统最大绝对误差
    err = new_err
    d = Melon_Network.train()
    new_err = d[0]
    print(d[3])
    # 将下面两行取消注释防止误差增大（但在训练过程中这是没有必要的）
    # if new_err - new > 0:
    #     break
# 显示神经网络如何使输入/出值与数值一一对应（解释器）
print(Melon_Network.interpreter)
# 对训练集测试训练结果
Melon_Network.test_list()
# 预测一个瓜的好坏（由解释器可知 好瓜 = 1，坏瓜 = 0）
Melon18 = Melon(tag=18,color='black' , root='shrink', sound='deep' , texture='blur'  , belly='wedged' , touch='slime')
res = Melon_Network.predict(Melon18)
# 例如预测结果约为 1，说明是好瓜
print(res)
print(Melon_Network)