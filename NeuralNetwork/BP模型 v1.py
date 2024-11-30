from math import exp
from random import random
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
    def __iter__(self):
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
def Prime_attach(s_v_prime : callable): # 装饰器，用于把一个单值函数和分别定义的它的导数合并
    # 语法为         @Prime_attach(导数名称)
    # 导数应提前定义 def 单值函数(单值变量): # 默认变量名为 variable
    # 其次定义函数体    函数体
    # 捆绑后的函数可以使用 Prime = True 作为参数调用导数
    # 例 @Prime_attach(delta) def sgn(x): return 0 if x < 0 else 1
    # 那么 sgn(0, Prime = True) 等价于 delta(0) 即调用了 delta 作为导数
    def attacher(single_valued : callable):
        def attached(variable, *, Prime = False):
            if Prime:
                return s_v_prime(variable)
            else:
                return single_valued(variable)
        return attached
    return attacher
def delta(x : float):
    return 1 if x == 0 else 0
@Prime_attach(delta)
def sgn(x : float):
    return 0 if x < 0 else 1
def sigmoidPrime(x : float): # sigmoid 的导数
    y = 1 / (1 + exp(-x))
    return y * (1 - y)
@Prime_attach(sigmoidPrime)
def sigmoid(x : float): # 神经网络中常用的激活函数
    return 1 / (1 + exp(-x))
def instances(init_ID): # 装饰器，返回一个类的装饰器和总字典，每个实例与一个ID绑定记录在总字典中
    # init_ID 为第一个实例的 ID，随后的实例 ID 递增 1
    static_identifier = init_ID
    full_instances = {}
    def register(object):
        nonlocal static_identifier, full_instances
        full_instances[static_identifier] = object
        static_identifier += 1
        return (static_identifier - 1)
    return register, full_instances
neuron_register, neurons = instances(-1) # 使用 neurons 存储所有已生成的神经元
layer_register, layers = instances(-1) # 使用 layers 存储所有已生成的层结构
activator_register, activators = instances(0)
activator_register(sgn)
activator_register(sigmoid) # 若需要内存优化，首先可以将 Neuron 中的 activator 改为 activator_id : int 并对全部代码做类似处理
class Neuron: # 单个神经元
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    activator_id : int # 激活函数 ID，sgn 为 0，sigmoid 为 1
    inputs : dict[int, float] # 输入和权重列表
    threshold : float # 阈值
    out : float # 储存输出值
    __slots__ = {'layer_id',
                 '__neuron_id',
                 'activator_id',
                 'inputs',
                 'threshold',
                 'out'}
    def __init__(self,
        activator_id : int = 1,
        inputs : dict[int, float] = dict(),
        threshold : float = random()) -> None:
        # 使用之前定义的注册器注册 ID，并加入到总字典 neurons
        self.__neuron_id = neuron_register(self)
        self.threshold = threshold
        self.activator_id = activator_id
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = {}
        self.out = 0
    def __str__(self) -> str: # 格式化可视化打印
        if self.inputs:
            n_str = '<Neuron {1} at {0}, activator = {2}, out = {3}, inputs = ['.format(
                hex(id(self)), self.__neuron_id, self.activator.__name__, self.out)
            for k, w in self.inputs.items():
                n_str += '{0} : {1}, '.format(k, w)
            n_str += '\b\b]>'
        else:
            n_str = '<Neuron {1} at {0}, activator = {2}, out = {3}>'.format(
                hex(id(self)), self.__neuron_id, activators[self.activator_id].__name__, self.out)
        return n_str
    def ID(self): # 返回自身 ID 值
        return self.__neuron_id
    def append(self, neuron, weight : float = 0.5): # 添加输入神经元
        if isinstance(neuron, Neuron):
            self.inputs[neuron.ID()] = weight
        elif isinstance(neuron, int):
            self.inputs[neuron] = weight
        else:
            pass
    def remove(self, neuron): # 移除某个输入神经元
        if isinstance(neuron, Neuron):
            del self.inputs[neuron.ID()]
        elif isinstance(neuron, int):
            del self.inputs[neuron]
        else:
            pass
    def __getitem__(self, input_id): # 得到对应输入神经元的权重
        if input_id in self.inputs.keys():
            return self.inputs[input_id]
        else:
            return None
    def __setitem__(self, neuron, weight): # 调整对应神经元的权重
        if isinstance(neuron, Neuron):
            if (input_id := neuron.ID()) in self.inputs.keys():
                self.inputs[input_id] = weight
        elif isinstance(neuron, int):
            if neuron in self.inputs.keys():
                self.inputs[neuron] = weight
        else:
            pass
    def __delitem__(self, neuron): # 移除一个输入神经元的链接
        if isinstance(neuron, Neuron):
            if (input_id := neuron.ID()) in self.inputs.keys():
                del self.inputs[input_id]
        elif isinstance(neuron, int):
            if neuron in self.inputs.keys():
                del self.inputs[neuron]
        else:
            pass
    def change_activator(self, activator_id : int): # 更改激活函数
        self.activator_id = activator_id
    def __call__(self, mode = 'normal', neuron_dict : dict = neurons) -> float: # 计算输出
        if self.inputs:
            var = -self.threshold
            for x, w in self.inputs.items():
                var += w * neuron_dict[x].out
            if mode == 'normal': # normal 将输出使用激活函数处理之后输出到 self.out
                self.out = activators[self.activator_id](var)
            elif mode == 'direct': # direct 将输出直接输出到 self.out
                self.out = var
            elif mode == 'derive': # derive 不赋值 self.out，只返回未处理的输出值
                return var
            return self.out
        else:
            pass
    def reset(self, threshold : float = 0.5) -> None: # 重设阈值
        self.threshold = threshold
    def read(self, datum) -> None: # 重设输出端口
        self.out = datum
Null = Neuron() # 空神经元
class Layer: # 神经网络中的一层结构
    __layer_id : int # 层 ID，作为私有属性不可更改
    neuron_dict : dict[int, Neuron] # 层内的神经元字典，可以根据 ID 查询
    __slots__ = {'__layer_id', 'neuron_dict'}
    def __init__(self, *neurons : Neuron) -> None:
        self.neuron_dict = {}
        self.__layer_id = layer_register(self) # 使用注册器注册 ID 并存储入总字典 layers
        for neuron in neurons:
            self.neuron_dict[neuron.ID()] = neuron
    def __str__(self) -> str: # 格式化打印
        if self.neuron_dict:
            l_str = '<Layer {1} at {0}, Neurons = ['.format(hex(id(self)), self.__layer_id)
            for n in self.neuron_dict.values():
                l_str += '\n    ' + str(n) + ','
            l_str += '\b]>'
        else:
            l_str = '<Layer {1} at {0}>'.format(hex(id(self)), self.__layer_id)
            return l_str
    def __getitem__(self, neuron_id : int) -> Neuron: # 简化代码，暂未使用
        return self.neuron_dict[neuron_id]
    def append(self, neuron : Neuron) -> None: # 层内增加神经元
        self.neuron_list[neuron.ID()] = neuron
    def remove(self, neuron : Neuron) -> None: # 层内移除神经元
        del self.neuron_dict[neuron.ID()]
    def neuron_ids(self) -> list[int]: # 作为连接层之间的辅助函数
        return list(self.neuron_dict.keys())
    def link_prev(self, prev_layer, *,
                  weight_key : callable = random,
                  **weight_args) -> None:
        # 连接两个层（将本层当作下一层），并计算 weight_key(weight_args) 作为连接权重
        if isinstance(prev_layer, Layer):
            prevs = prev_layer.neuron_ids()
            l = len(prevs)
            for i, n in self.neuron_dict.items():
                w = [(weight_key() if not weight_args else weight_key(**weight_args)) for t in range(l)]
                n.inputs = dict(zip(prevs, w))
        else:
            pass
    def link_next(self, next_layer) -> None: # 连接两个层（将本层当作上一层）
        if isinstance(next_layer, Layer):
            next_layer.link_prev(self)
        else:
            pass
    def __call__(self) -> None: # 本层所有神经元计算处理输出值
        for n in self.neuron_dict.values():
            n()
    def __len__(self) -> int: # 本层规模，即本层包含神经元数量
        return len(self.neuron_dict)
    def ID(self) -> int: # 返回本层 ID
        return self.__layer_id
    def change_activator(self, activator_id : int): # 更改本层全体神经元的激活函数
        for n in self.neuron_dict.values():
            n.change_activator(activator_id)
Null_layer = Layer(Null) # 空层
def spawn_layer(number_of_neurons : int,
                layer_activator_id : int = 1,
                threshold_key : callable = random,
                **threshold_args) -> Layer:
    # 使用格式化手段生成一个包含 number_of_neurons 个神经元的层，少于 1 个则返回空层
    # 每个神经元的激活函数的 ID 都是 layer_activator_id，阈值通过 threshold_key(threshold_args) 计算
    if number_of_neurons <= 0:
        return Null_layer
    else:
        l = Layer()
        for i in range(number_of_neurons):
            n = Neuron(layer_activator_id, dict(),
                       threshold_key() if not threshold_args else threshold_key(**threshold_args))
            l.neuron_dict[n.ID()] = n
        return l
class Network: # 神经网络（基础类型）
    layers_struct : dict # 层次结构，如 {0: [1,2]} 表示 ID 为 1，2 的层的输出层是 ID 为 0 的层
    data : Data # 神经网络使用的数据集
    input_layer : Layer # 输入层
    input_feats : list # 输入属性，可以无定义
    interpreter : dict # 解释字典，__in 解释每个输入神经对应的属性，__out 解释每个输出神经对应的属性
                       # 并将每个属性取值解释为一个数
    output_layer : Layer # 输出层
    output_feats : list # 输出属性，可以无定义
    full_layers : dict # 在运算时和训练时调用的 层结构的总字典
    update_order : list # 通过 init 方法生成的计算顺序，决定层的输出顺序
    learning_rate : float # 学习率
    def set_input_layer(self, input_dimension : int,
                        activator_id : int = 1) -> Layer: # 设置输入层
        if input_dimension > 0:
            i = spawn_layer(input_dimension, activator_id, sgn, variable = -1)
            self.input_layer = i
            self.layers_struct[i.ID()] = []
            return i
    def set_output_layer(self, output_dimension : int,
                         prev_layer_id : int,
                         activator_id : int = 1) -> Layer: # 设置输出层
        if output_dimension > 0:
            o = spawn_layer(output_dimension, activator_id)
            self.output_layer = o
            self.layers_struct[o.ID()] = []
            o.link_prev(self.full_layers[prev_layer_id])
            self.layers_struct[prev_layer_id].append(o.ID())
            return o
    def add_hidden_layer_to(self, dimension : int,
                            prev_layer_id : int,
                            activator_id : int = 1) -> Layer: # 增加隐藏层
        if dimension > 0:
            h = spawn_layer(dimension, activator_id)
            h.link_prev(self.full_layers[prev_layer_id])
            self.layers_struct[h.ID()] = []
            self.layers_struct[prev_layer_id].append(h.ID())
            return h
    def link_to(self, below : Layer, above : Layer) -> None: # 连接两个层，暂未使用
        above.link_prev(below)
        self.layers_struct[below.ID()].append(above.ID())
    def __init__(self,input_dimension : int = 1,
                 output_dimension : int = 1,
                 activator_id : int = 1,
                 hidden_dimensions : list = [][:], *,
                 learning_rate = 0.02,
                 full_layers = layers) -> None:
        self.layers_struct = {}
        self.interpreter = {}
        self.learning_rate = learning_rate
        self.full_layers = full_layers
        i = self.set_input_layer(input_dimension, activator_id)
        if hidden_dimensions:
            v = i.ID()
            for d in hidden_dimensions:
                p = self.add_hidden_layer_to(d, v, activator_id)
                v = p.ID()
        else:
            v = i.ID()
        self.set_output_layer(output_dimension, v, activator_id)
    def __str__(self) -> str: # 格式化打印
        n_str = '<Network object at {0}, Layers = ['.format(hex(id(self)))
        for layer_id in self.layers_struct.keys():
            n_str += '\n  ' + str(self.full_layers[layer_id]) + ','
        n_str += '\b]'
        if self.interpreter:
            n_str += str(self.interpreter)
        n_str += '>'
        return n_str
    def load_data(self, data: Data,
                  input_feats : list = [][:],
                  output_feats : list = [][:]) -> None: # 加载数据集
        self.data = data
        if input_feats:
            if len(input_feats) == len(self.input_layer): # 设置输入属性，必须和输入层对齐
                self.input_feats = input_feats
        if output_feats:
            if len(output_feats) == len(self.output_layer): # 设置输出属性，必须和输出层对齐
                self.output_feats = output_feats
    def set_up_feats(self,
                     input_feats : list = [][:],
                     output_feats : list = [][:]) -> None: # 设置输入/出属性，必须和输入/出层对齐
        if input_feats:
            if len(input_feats) == len(self.input_layer):
                self.input_feats = input_feats
        if output_feats:
            if len(output_feats) == len(self.output_layer):
                self.output_feats = output_feats
    def init(self, compile_mode = 'compile') -> None: # 选择解释字典模式下初始化加载模式和输出顺序
        if self.data:
            for feat in self.data.feats:
                d = self.data.feat_dict(feat, mode = compile_mode)
                if d:
                    self.interpreter[feat] = d
            if self.input_feats:
                self.interpreter['__in'] = dict(zip(self.input_feats, self.input_layer.neuron_dict.keys()))
            if self.output_feats:
                self.interpreter['__out'] = dict(zip(self.output_feats, self.output_layer.neuron_dict.keys()))
        self.init_order()
    def init_order(self) -> None: # 要求 output 层只有一个前接层，要求无循环层，初始化输出顺序
        l = [self.input_layer.ID()]
        m = 0
        v = self.output_layer.ID()
        while True:
            if l[m] != v:
                p = self.layers_struct[l[m]]
                l += p
                m += 1
            else:
                break
        self.update_order = l
    def __call__(self) -> None: # 按输出顺序依次计算处理每层的输出值
        if self.update_order:
            for i in range(len(self.update_order)):
                self.full_layers[self.update_order[i]]()
    def read_data(self, item_id) -> None: # 输入层读入一条数据并更新每层的输出值
        if item_id < len(self.data):
            t = self.data[item_id]
            i = self.input_layer
            for feat in self.input_feats:
                n = self.interpreter['__in'][feat]
                if feat in self.interpreter.keys():
                    i[n].read(self.interpreter[feat][t[feat]])
                else:
                    i[n].read(t[feat])
            self()
    def predict(self, item) -> dict: # 根据一条数据计算输出
        t = item
        i = self.input_layer
        for feat in self.input_feats:
            n = self.interpreter['__in'][feat]
            if feat in self.interpreter.keys():
                i[n].read(self.interpreter[feat][t[feat]])
            else:
                i[n].read(t[feat])
        self()
        d = {feat: self.output_layer[self.interpreter['__out'][feat]].out for feat in self.output_feats}
        return d
    def test(self, item_id): # 将数据集中的一条数据与输出比较测试
        p = self.predict(self.data.items[item_id])
        y = {}
        for feat in self.output_feats:
            t = self.data.items[item_id][feat]
            y[feat] = self.interpreter[feat][t]
        print('\nTesting:   ', item_id)
        print('Predicted:', p)
        print('Observed: ', y,'\n')
    def test_list(self, item_id_list = [][:]): # 测试一个数据集，若 item_id_list 为空则测试整个数据集
        i = item_id_list if item_id_list else range(len(self.data))
        for datum in i:
            self.test(datum)
class BP_Network(Network): # Back Propagation 神经网络，结构为 输入层 - 隐藏层 - 输出层
    data : Data
    input_layer : Layer
    input_feats : list
    hidden_layer : Layer # 与神经网络不同，只有一层隐藏层，因此不需要输出顺序和层总字典
    interpreter : dict
    output_layer : Layer
    output_feats : list
    learning_rate : float
    __slots__ = {'data', 'input_layer', 'hidden_layer', 'output_layer', 'interpreter',
                 'input_feats', 'output_feats', 'learning_rate'}
    def set_input_layer(self, input_dimension : int,
                        activator_id : int = 1) -> Layer: # 设置输入层
        if input_dimension > 0:
            i = spawn_layer(input_dimension, activator_id, sgn, variable = -1)
            self.input_layer = i
            return i
    def set_output_layer(self, output_dimension : int,
                         activator_id : int = 1) -> Layer: # 设置输出层
        if output_dimension > 0 and self.hidden_layer:
            o = spawn_layer(output_dimension, activator_id)
            self.output_layer = o
            o.link_prev(self.hidden_layer)
            return o
    def set_hidden_layer(self, hidden_dimension : int,
                        activator_id : int = 1) -> Layer: # 设置隐藏层
        if hidden_dimension > 0 and self.input_layer:
            h = spawn_layer(hidden_dimension, activator_id)
            self.hidden_layer = h
            h.link_prev(self.input_layer)
            return h
    def __init__(self, input_dimension : int = 1, # 输入层规模
                 hidden_dimension : int = 1, # 隐藏层规模
                 output_dimension : int = 1, # 输出层规模
                 activator_id : int = 1, *, # 神经网络总体使用的激活函数 ID
                 learning_rate = 0.02) -> None: # 学习率
        self.interpreter = {}
        self.learning_rate = learning_rate
        self.set_input_layer(input_dimension, activator_id)
        self.set_hidden_layer(hidden_dimension, activator_id)
        self.set_output_layer(output_dimension, activator_id)
    def __str__(self) -> str: # BP 网络的格式化打印
        bp_n_str = '<Back Propagation Network at {0}, Input = {1}, Hidden = {2}, Output = {3}>'.format(
            hex(id(self)), str(self.input_layer), str(self.hidden_layer), str(self.output_layer))
        if self.interpreter:
            bp_n_str += '\nInterprets: ' + str(self.interpreter)
        return bp_n_str
    def init(self, compile_mode = 'compile') -> None: # 简化的初始化函数
        if self.data:
            for feat in self.data.feats:
                d = self.data.feat_dict(feat, mode = compile_mode)
                if d:
                    self.interpreter[feat] = d
            if self.input_feats:
                self.interpreter['__in'] = dict(zip(self.input_feats, self.input_layer.neuron_dict.keys()))
            if self.output_feats:
                self.interpreter['__out'] = dict(zip(self.output_feats, self.output_layer.neuron_dict.keys()))
    def __call__(self) -> None: # 分别激活隐藏层和输出层
        (self.hidden_layer)()
        (self.output_layer)()
    def read_data(self, item_id : int) -> float: # 读取一条数据并随后更新网络的
        # 读取一条数据
        t = self.data[item_id]
        i = self.input_layer
        i_m = i.neuron_dict.items()
        for feat in self.input_feats:
            n = self.interpreter['__in'][feat]
            if feat in self.interpreter.keys():
                i[n].read(self.interpreter[feat][t[feat]])
            else:
                i[n].read(t[feat])
        self()
        gradient = {} # 计算输出层梯度项
        E = 0 # E 返回本项数据的方均误差
        o = self.output_layer
        o_m = o.neuron_dict.items()
        for feat in self.output_feats:
            n = self.interpreter['__out'][feat]
            p = o[n].out
            f_p = activators[o[n].activator_id]
            if feat in self.output_feats:
                y = self.interpreter[feat][t[feat]]
            else:
                y = t[feat]
            E += (y - p) ** 2
            if f_p == sigmoid:
                gradient[n] = p * (1 - p) * (y - p)
            else:
                v = o[n](mode = 'derive')
                gradient[n] = (y - p) * f_p(v, Prime = True)
        e_h = {} # 计算隐藏层梯度项
        h = self.hidden_layer
        h_m = h.neuron_dict.items()
        for h_i, h_n in h_m:
            if (f_p := activators[h_n.activator_id]) == sigmoid:
                s = sum(o_n.inputs[h_i] * gradient[o_i] for o_i, o_n in o_m)
                m = h_n.out
                e_h[h_i] = m * (1 - m) * s
            else:
                v = h_n(mode = 'derive')
                e_h[h_i] = sum(o_n.inputs[h_i] * gradient[o_i] * f_p(v, Prime = True) for o_i, o_n in o_m)
        for o_i, o_n in o_m: # 更新输出层连接权和阈值
            for h_i, h_n in h_m:
                o_n.inputs[h_i] += self.learning_rate * h_n.out * gradient[o_i]
                o_n.threshold -= self.learning_rate * gradient[o_i]
        for h_i, h_n in h_m: # 更新隐藏层连接权和阈值
            for i_i, i_n in i_m:
                h_n.inputs[i_i] += self.learning_rate * i_n.out * e_h[h_i]
                h_n.threshold -= self.learning_rate * e_h[h_i]
        return E / 2
    def train(self, train_list : list = [][:]) -> list:
        # 根据数据的 ID 列表 train_list 训练，若列表为空则根据整个数据集训练
        # 返回一个列表，第一项为累积误差，第二项为方均误差列表，第三项为最大的方均误差，最后为系统最大绝对误差
        E = []
        l = train_list if train_list else range(len(self.data))
        for datum in l:
            e = self.read_data(datum)
            # print(self.output_layer)
            E.append(e)
            Q = sum(E) / len(E)
        return [Q, E, max(E), (2 * max(E)) ** .5]

##################################################### 一个例子
# 设置神经网络基础结构
Melon_Network = BP_Network(input_dimension = 6,
                           hidden_dimension = 6,
                           output_dimension = 1,
                           activator_id = 1,
                           learning_rate = 5e-2)
# 训练使用的西瓜的数据集
Melons_data = Data(
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
# 神经网络加载数据集
Melon_Network.load_data(Melons_data,
    ['color', 'root', 'sound', 'texture', 'belly', 'touch'], # 待输入的属性，对齐 6 个输入神经元
    ['quality']) # 待预测的属性，对齐 1 个输出神经元
Melon_Network.init('normal') # 初始化
M = 50000 # 训练总次数
new_err = 0
for i in range(1): # 训练 1 次更新 err 项
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