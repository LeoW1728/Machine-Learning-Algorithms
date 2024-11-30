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
from math import exp
from random import random
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
        attached.__name__ = single_valued.__name__
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
def sigmoid(x : float): # 老牌神经网络中常用的激活函数
    return 1 / (1 + exp(-x))
@Prime_attach(lambda x : 1 if x > 0 else 0)
def ReLU(x : float): # 更常用的线性激活函数，收敛速度更快
    return x if x > 0 else 0
def Gaussian_kernel(r : float, sigma : float, *, Prime : bool = False):
    # RBF 网络使用的激活函数，r 是径的模
    if not Prime:
        return 0 if sigma == 0 else exp(- (r ** 2) / (2 * sigma ** 2))
    else:
        return 0 if sigma == 0 else -exp(- (r ** 2) / (2 * sigma ** 2)) * r / sigma ** 2
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
activator_register, activators = instances(0) # 使用 activators 存储所有可能被调用的激活函数
activator_register(sgn)
activator_register(sigmoid)
activator_register(ReLU)
activator_register(Gaussian_kernel)
class Neuron:
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重列表
    out : float # 储存输出值
    def __init__(self):
        self.__neuron_id = neuron_register(self)
    def ID(self): # 返回自身ID值
        return self.__neuron_id
    def read(self, datum) -> None: # 重设输出端口
        self.out = datum
    def __getitem__(self, input_id) -> float: # 得到对应输入神经元的权重
        if input_id in self.inputs.keys():
            return self.inputs[input_id]
        else:
            return 0.
    def __setitem__(self, neuron, weight) -> None: # 调整对应神经元的权重
        if isinstance(neuron, Neuron):
            if (input_id := neuron.ID()) in self.inputs.keys():
                self.inputs[input_id] = weight
        elif isinstance(neuron, int):
            if neuron in self.inputs.keys():
                self.inputs[neuron] = weight
    def __delitem__(self, neuron) -> None: # 移除一个输入神经元的链接
        if isinstance(neuron, Neuron):
            if (input_id := neuron.ID()) in self.inputs.keys():
                del self.inputs[input_id]
        elif isinstance(neuron, int):
            if neuron in self.inputs.keys():
                del self.inputs[neuron]
Null = Neuron() # 空神经元
class BP_Neuron(Neuron): # 单个神经元
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重列表
    threshold : float # 阈值
    out : float # 储存输出值
    __slots__ = {'layer_id',
                 '__neuron_id',
                 'activator_id',
                 'inputs',
                 'threshold',
                 'out'}
    def __init__(self,
        inputs : dict[int, float] = dict(),
        threshold : float = random()) -> None:
        # 使用之前定义的注册器注册 ID，并加入到总字典 neurons
        self.__neuron_id = neuron_register(self)
        self.threshold = threshold
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = {}
        self.out = 0
    def ID(self): # 返回自身ID值
        return self.__neuron_id
    def __str__(self) -> str: # 格式化可视化打印
        if self.inputs:
            n_str = '<BP Neuron {1} at {0}, out = {2}, inputs = ['.format(
                hex(id(self)), self.__neuron_id, self.out)
            for k, w in self.inputs.items():
                n_str += '{0} : {1}, '.format(k, w)
            n_str += '\b\b]>'
        else:
            n_str = '<BP Neuron {1} at {0}, out = {2}>'.format(
                hex(id(self)), self.__neuron_id, self.out)
        return n_str
    def append(self, neuron, weight : float = 0.5): # 添加输入神经元
        if isinstance(neuron, Neuron):
            self.inputs[neuron.ID()] = weight
        elif isinstance(neuron, int):
            self.inputs[neuron] = weight
    def remove(self, neuron): # 移除某个输入神经元
        if isinstance(neuron, Neuron):
            del self.inputs[neuron.ID()]
        elif isinstance(neuron, int):
            del self.inputs[neuron]
    def __call__(self, activator_id : int = 1, mode = 'normal') -> float: # 计算输出
        if self.inputs:
            var = -self.threshold
            for x, w in self.inputs.items():
                var += w * neurons[x].out
            if mode == 'normal': # normal 将输出使用激活函数处理之后输出到 self.out
                self.out = activators[activator_id](var)
            elif mode == 'direct': # direct 将输出直接输出到 self.out
                self.out = var
            elif mode == 'derive': # derive 不赋值 self.out，只返回未处理的输出值
                return var
            return self.out
    def reset(self, threshold : float = 0.5) -> None: # 重设阈值
        self.threshold = threshold
class RBF_Neuron(Neuron):
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重值列表
    centers : dict # 输入和中心值列表
    kernel_width : float # 高斯核宽度
    out : float # 储存输出值
    __slots__ = {'layer_id',
                 '__neuron_id',
                 'activator_id',
                 'inputs',
                 'centers',
                 'kernel_width',
                 'out'}
    def __init__(self,
        inputs : dict = dict(), centers : dict = dict(),
        kernel_width : float = 1.) -> None:
        # 使用之前定义的注册器注册 ID，并加入到总字典 neurons
        self.__neuron_id = neuron_register(self)
        self.kernel_width = kernel_width
        self.inputs = inputs
        self.centers = centers
        self.out = 0
    def ID(self): # 返回自身ID值
        return self.__neuron_id
    def __str__(self) -> str: # 格式化可视化打印
        if self.inputs:
            n_str = '<RBF Neuron {1} at {0}, out = {2}, width = {3}, inputs = ['.format(
                hex(id(self)), self.__neuron_id, self.out, self.kernel_width)
            for k in self.inputs.keys():
                n_str += '{0} : {1} | {2}, '.format(k, self.inputs[k], self.centers[k])
            n_str += '\b\b]>'
        else:
            n_str = '<RBF Neuron {1} at {0}, out = {2}>'.format(
                hex(id(self)), self.__neuron_id, self.out)
        return n_str
    def append(self, neuron, weight : float = 0.5, center : float = 0.): # 添加输入神经元
        if isinstance(neuron, Neuron):
            n = neuron.ID()
            self.inputs[n] = weight
            self.centers[n] = center
        elif isinstance(neuron, int):
            self.inputs[neuron] = weight
            self.centers[neuron] = center
    def __call__(self, mode = 'normal') -> list[float]: # 计算输出
        if self.inputs and self.centers:
            if mode == 'normal': # normal 将输出使用激活函数处理之后输出到 self.out
                var, s = 0, self.kernel_width
                for i in self.inputs.keys():
                    f = neurons[i].out - self.centers[i]
                    var += self.inputs[i] * Gaussian_kernel(f, s)
                    self.out = var
                return [var]
            elif mode == 'weight': # weight 返回权重值辅助列表
                s = self.kernel_width
                return [Gaussian_kernel(neurons[i].out - self.centers[i],
                            s) for i in self.inputs.keys()]
            elif mode == 'center': # center 返回中心值辅助列表
                s = self.kernel_width
                return [self.inputs[i] * Gaussian_kernel(neurons[i].out - self.centers[i],
                            s, Prime = True) for i in self.inputs.keys()]
            elif mode == 'derive': # derive 返回输出值辅助列表
                s = self.kernel_width
                return [self.inputs[i] * Gaussian_kernel(neurons[i].out - self.centers[i],
                            s) for i in self.inputs.keys()]
            return 
    def reset(self, kernel_width : float = 0.5) -> None: # 重设阈值
        self.kernel_width = kernel_width
class Layer: # 神经网络中的一层结构
    __layer_id : int # 层 ID，作为私有属性不可更改
    neuron_dict : dict # 层内的神经元字典，可以根据 ID 查询
    __layer_type : str # 层类型，比如普通神经网络 BP 层，循环 R 层，径向基 RBF 层等，作为私有属性不可更改
    __slots__ = {'__layer_id', 'neuron_dict', '__layer_type'}
    def __init__(self, type : str = 'BP', *neurons : Neuron) -> None:
        self.__layer_type = type
        self.neuron_dict = {}
        self.__layer_id = layer_register(self) # 使用注册器注册 ID 并存储入总字典 layers
        for neuron in neurons:
            self.neuron_dict[neuron.ID()] = neuron
    def __str__(self) -> str: # 格式化打印
        if self.__layer_id == -1:
            return '<Null Layer at {0}>'.format(hex(id(self)))
        if self.neuron_dict:
            l_str = '<{2} Layer {1} at {0}, Neurons = ['.format(hex(id(self)),
                                                        self.__layer_id, self.__layer_type)
            for n in self.neuron_dict.values():
                l_str += '\n    ' + str(n) + ','
            l_str += '\b]>'
        else:
            l_str = '<{2} Layer {1} at {0}>'.format(hex(id(self)),
                                            self.__layer_id, self.__layer_type)
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
                  key : callable = random,
                  **args) -> None:
        # 连接两个层（将本层当作下一层），并计算 weight_key(weight_args) 作为连接权重
        if isinstance(prev_layer, Layer):
            prevs = prev_layer.neuron_ids()
            l = len(prevs)
            if self.__layer_type == 'BP':
                for n in self.neuron_dict.values():
                    w = [(key() if not args else key(**args)) for t in range(l)]
                    n.inputs = dict(zip(prevs, w))
            elif self.__layer_type == 'RBF':
                c_aux = []
                for n in prev_layer.neurons():
                    c_aux.append(n.out)
                m, M, p= min(c_aux), max(c_aux), len(c_aux)
                for n in self.neurons():
                    w = [(key() if not args else key(**args)) for t in range(l)]
                    # c = m + (M - m) * (.5 + j) / p
                    # j += 1
                    c = [m + (M - m) * (.5 + j) / p for j in range(p)]
                    n.reset((sum((c_aux[j] - c[j]) ** 2 for j in range(p)) / p) ** .5)
                    n.inputs = dict(zip(prevs, w))
                    # n.centers = {prev_i : c for prev_i in prevs}
                    n.centers = dict(zip(prevs, c))
    def link_next(self, next_layer) -> None: # 连接两个层（将本层当作上一层）
        if isinstance(next_layer, Layer):
            next_layer.link_prev(self)
        else:
            pass
    def init(self): # 随机预设本层所有神经元的值
        for n in self.neurons():
            n.out = random()
    def __call__(self, mode = 'normal', activator_id : int = 1) -> None: # 本层所有神经元计算处理输出值
        if self.__layer_type == 'BP':
            for n in self.neurons():
                n(activator_id, mode)
        elif self.__layer_type == 'RBF':
            for n in self.neurons():
                n(mode)
    def __len__(self) -> int: # 本层规模，即本层包含神经元数量
        return len(self.neuron_dict)
    def ID(self) -> int: # 返回本层 ID
        return self.__layer_id
    def neurons(self): # 简化代码
        return self.neuron_dict.values()
    def items(self): # 简化代码
        return self.neuron_dict.items()
    def type(self): # 返回本层 type
        return self.__layer_type
Null_layer = Layer(Null) # 空层
def spawn_BP_layer(number_of_neurons : int,
                threshold_key : callable = random,
                **threshold_args) -> Layer:
    # 使用格式化手段生成一个包含 number_of_neurons 个神经元的层，少于 1 个则返回空层
    # 每个神经元的激活函数的 ID 都是 layer_activator_id，阈值通过 threshold_key(threshold_args) 计算
    if number_of_neurons <= 0:
        return Null_layer
    else:
        l = Layer('BP')
        for i in range(number_of_neurons):
            n = BP_Neuron(dict(),
                    threshold_key() if not threshold_args else threshold_key(**threshold_args))
            l.neuron_dict[n.ID()] = n
        return l
def spawn_RBF_layer(number_of_neurons : int) -> Layer:
    if number_of_neurons <= 0:
        return Null_layer
    else:
        l = Layer('RBF')
        for i in range(number_of_neurons):
            n = RBF_Neuron()
            l.neuron_dict[n.ID()] = n
        return l
def spawn_layer(number_of_neurons : int, layer_type : str = 'BP',
                key : callable = random, **args) -> Layer:
    if layer_type == 'BP':
        return spawn_BP_layer(number_of_neurons, key, **args)
    elif layer_type == 'RBF':
        return spawn_RBF_layer(number_of_neurons)
class Network: # 多层神经网络，结构为 输入层 - 隐藏层 - 输出层
    # 当前可以使用的层类型有
    # 'BP' 为 Back Propagation 网络层结构
    #     使用任意的激活函数，每个神经元除权重列表外还有阈值
    # 'RBF' 为 Radial Basis Function 网络层结构
    #     使用线性欧式距离表示径距，使用高斯核作为径向基，
    #     每个神经元除权重列表外还有变量中心列表和核宽度
    data : Data
    Layer_ids : list[int] # 层顺序
    input_feats : list
    interpreter : dict
    output_feats : list
    learning_rate : float
    activator_id : int
    gradients : dict # 储存梯度信息，用于递归计算
    __slots__ = {'data', 'interpreter', 'input_feats', 'output_feats',
                 'learning_rate', 'activator_id', 'Layer_ids', 'gradients'}
    def set_input_layer(self, input_dimension : int) -> Layer: # 设置输入层
        if input_dimension > 0:
            i = spawn_layer(input_dimension, 'BP', sgn, variable = -1)
            i.init()
            return i
    def set_output_layer(self, output_dimension : int, prev_layer : Layer,
                         l_t : str = 'BP') -> Layer: # 设置输出层
        if output_dimension > 0 and prev_layer:
            o = spawn_layer(output_dimension, l_t)
            o.link_prev(prev_layer)
            o()
            return o
    def set_hidden_layer(self, hidden_dimension : int, prev_layer : Layer,
                         l_t : str = 'BP') -> Layer: # 设置隐藏层
        if hidden_dimension > 0 and prev_layer:
            h = spawn_layer(hidden_dimension, l_t)
            h.link_prev(prev_layer)
            h()
            return h
    def __init__(self, activator_id : int = 1, # 神经网络总体使用的激活函数 ID
                 learning_rate = 0.02, # 学习率
                 input_dimension : int = 1, # 输入层类型和规模
                 output_layer : tuple[str, int] = ('BP', 1), # 输出层类型和规模
                 hidden_layers : list[tuple[str, int]] = [][:]) -> None: # 隐藏层类型和规模
        # 例如使用 Network(1, .02, 3, ('RBF', 4), [('BP', 5), ('RBF', 6)])
        # 这将生成一个神经网络，结构为
        #     输入层(3个神经元) -> 隐藏层(BP, 5个神经元)
        #         -> 隐藏层(RBF，6个神经元) - 输出层(RBF，4个神经元)
        self.interpreter = {}
        self.learning_rate = learning_rate
        self.activator_id = activator_id
        p = self.set_input_layer(input_dimension)
        self.Layer_ids = [p.ID()]
        if hidden_layers:
            for l_t, h_d in hidden_layers:
                p = self.set_hidden_layer(h_d, p, l_t)
                self.Layer_ids.append(p.ID())
        p = self.set_output_layer(output_layer[1], p, output_layer[0])
        self.Layer_ids.append(p.ID())
        self.gradients = {l:{} for l in self.Layer_ids}
    def __str__(self) -> str: # BP 网络的格式化打印
        layers_str = ''
        for l_i in self.Layer_ids:
            layers_str += '\n  ' + str(layers[l_i]) + ','
        layers_str += '\b'
        bp_n_str = '<Back Propagation Network at {0} using activator {1}, Layers = [{2}]>'.format(
            hex(id(self)), activators[self.activator_id].__name__, layers_str)
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
            layers[l_i](mode, self.activator_id)
    def gradient(self, t : Item) -> float: # 根据一项数据更新梯度值
        E = 0
        D, I = self.Layer_ids, self.interpreter
        d, f_p = len(D) - 1, activators[self.activator_id]
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
        while d >= 0:
            # p representing post layer, l representing layer(now)
            # d = dictionary(gradients), m = items, i = id, n = neuron
            p, l = l, layers[D[d]]
            p_d, l_d = l_d, self.gradients[D[d]]
            p_m, l_m = l_m, l.items()
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
        d, f_p = len(D) - 1, activators[self.activator_id]
        l, l_d = layers[D[d]], self.gradients[D[d]]
        l_m = l.items()
        d -= 1
        while d >= 0:
            p, l = l, layers[D[d]]
            p_d, l_d = l_d, self.gradients[D[d]]
            p_m, l_m = l_m, l.items()
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
                        # 若使用此段代码，则将更新 RBF层 神经元核宽度
                        # qc = r * p_d[p_i] * Gaussian_kernel(
                        #     l_n.out - p_n.centers[l_i], p_n.kernel_width)
                        # qs = p_n.inputs[l_i] * (l_n.out - p_n.centers[l_i]) ** 2 / p_n.kernel_width ** 3
                        # p_n.kernel_width += qc * qs
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
            if feat in self.interpreter.keys():
                i[n].read(self.interpreter[feat][t[feat]])
            else:
                i[n].read(t[feat])
        self()
        d = {feat: layers[D[len(D) - 1]][self.interpreter['__out'][feat]].out for feat in self.output_feats}
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
Melon_Network = Network(activator_id = 1,
                           learning_rate = .02,
                           input_dimension = 6,
                           output_layer = ('BP', 1),
                           hidden_layers = [('BP', 6),('RBF', 6)])
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
# print(Melon_Network)
M = 5000 # 训练总次数
new_err = 0
for i in range(1): # 训练 1 次更新 err 项
    err = new_err
    d = Melon_Network.train()
    new_err = d[0]
for i in range(M): 
    err = new_err
    d = Melon_Network.train()
    new_err = d[0]
    # 将下面一行取消注释可以显示目前网络在训练集里的最大绝对误差
    print(d[3])
    # 将下面两行取消注释防止误差增大（但在训练过程中这是没有必要的）
    # if new_err - new > 0:
    #     break
# 显示神经网络如何使输入/出值与数值一一对应（解释器）
print(Melon_Network.interpreter)
# 对训练集测试训练结果
# Melon_Network.test_list()
# 预测一个瓜的好坏（由解释器可知瓜的好坏与哪个数字对应，默认在 0 和 1之间选择）
Melon18 = Melon(tag=18,color='black' , root='shrink', sound='deep' , texture='blur'  , belly='wedged' , touch='slime')
res = Melon_Network.predict(Melon18)
# 例如预测结果为 0.9，约为 1，根据解释器 'quality': {'good': 1, 'bad': 0} 说明是好瓜
print(res)