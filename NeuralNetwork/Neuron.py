from math import exp, tanh, cosh
from random import random
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
def ReLU(x : float): # CNN 网络常用的线性激活函数，收敛速度更快
    return x if x > 0 else 0
@Prime_attach(lambda x : (1 / cosh(x)) ** 2)
def Tanh(x : float): # RNN 网络常用的激活函数
    return tanh(x)
def Gaussian_kernel(r : float, sigma : float, *, Prime : bool = False):
    # RBF 网络使用的激活函数，r 是径的模
    if not Prime:
        return 0 if sigma == 0 else exp(- (r ** 2) / (2 * sigma ** 2))
    else:
        return 0 if sigma == 0 else -exp(- (r ** 2) / (2 * sigma ** 2)) * r / sigma ** 2
activator_register(sgn)
activator_register(sigmoid)
activator_register(ReLU)
activator_register(Tanh)
activator_register(Gaussian_kernel)
class Neuron:
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重列表
    out : float # 储存输出值
    def __init__(self):
        self.__neuron_id = neuron_register(self)
    def __str__(self):
        return '<Neuron at {0}, out = {1}>'.format(hex(id(self)), self.out)
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
        else:
            pass
    def __delitem__(self, neuron) -> None: # 移除一个输入神经元的链接
        if isinstance(neuron, Neuron):
            if (input_id := neuron.ID()) in self.inputs.keys():
                del self.inputs[input_id]
        elif isinstance(neuron, int):
            if neuron in self.inputs.keys():
                del self.inputs[neuron]
        else:
            pass
Null = Neuron() # 空神经元
class BP_Neuron(Neuron): # 单个 全连接网络 神经元
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重列表
    threshold : float # 阈值
    out : float # 储存输出值
    __slots__ = {'__neuron_id',
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
        else:
            pass
    def remove(self, neuron): # 移除某个输入神经元
        if isinstance(neuron, Neuron):
            del self.inputs[neuron.ID()]
        elif isinstance(neuron, int):
            del self.inputs[neuron]
        else:
            pass
    def __call__(self, mode = 'normal', activator_id = 1) -> float: # 计算输出
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
        else:
            pass
    def reset(self, threshold : float = 0.5) -> None: # 重设阈值
        self.threshold = threshold
class RBF_Neuron(Neuron): # 单个 RBF 网络 神经元
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重值列表
    centers : dict # 输入和中心值列表
    kernel_width : float # 高斯核宽度
    out : float # 储存输出值
    __slots__ = {'__neuron_id',
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
        else:
            pass
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
        else:
            pass
    def reset(self, kernel_width : float = 0.5) -> None: # 重设阈值
        self.kernel_width = kernel_width
class RNN_Neuron(Neuron): # 单个 RNN 网络 神经元
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重列表
    out : float # 储存输出值
    memory : float # 储存 上一次 输出值
    memory_inputs : dict # 对自身循环层的输入和权重列表
    __slots__ = {'__neuron_id',
                 'inputs',
                 'out',
                 'memory',
                 'memory_inputs'}
    def __init__(self) -> None:
        # 使用之前定义的注册器注册 ID，并加入到总字典 neurons
        self.memory = 0.
        self.out = 0.
        self.inputs = {}
        self.memory_inputs = {}
        self.__neuron_id = neuron_register(self)
    def ID(self): # 返回自身ID值
        return self.__neuron_id
    def __str__(self) -> str: # 格式化可视化打印
        if self.inputs:
            n_str = '<RNN Neuron {1} at {0}, out = {2}, last out = {3}, inputs = ['.format(
                hex(id(self)), self.__neuron_id, self.out, self.memory)
            for k, w in self.inputs.items():
                n_str += '{0} : {1}, '.format(k, w)
            for k, w in self.memory_inputs.items():
                n_str += '{0} : {1}, '.format(k, w)
            n_str += '\b\b]>'
        else:
            n_str = '<RNN Neuron {1} at {0}, out = {2}, last_out = {3}>'.format(
                hex(id(self)), self.__neuron_id, self.out, self.memory)
        return n_str
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
    def __call__(self, mode = 'normal', activator_id = 1) -> float: # 计算输出
        if self.inputs:
            var = 0
            for x, w in self.inputs.items():
                var += w * neurons[x].out
            for x, w in self.memory_inputs.items():
                var += w * neurons[x].memory
            if mode == 'normal': # normal 将输出使用激活函数处理之后输出到 self.out
                self.out = activators[activator_id](var)
            elif mode == 'direct': # direct 将输出直接输出到 self.out
                self.out = var
            elif mode == 'derive': # derive 不赋值 self.out，只返回未处理的输出值
                return var
            return self.out
        else:
            pass
    def memorize(self): # 记录本轮输出作为“上一轮输出”
        self.memory = self.out
class LSTM_Neuron(Neuron): # 单个 LSTM 网络 神经元
    __neuron_id : int # 本身的 ID，作为私有变量不可修改
    inputs : dict # 输入和权重列表
    out : float # 储存输出值
    memory : float # 储存 上一次 输出值
    memory_inputs : dict # 对自身循环层的输入和权重列表
    __slots__ = {'__neuron_id',
                 'inputs',
                 'out',
                 'memory',
                 'memory_inputs'}
    def __init__(self) -> None:
        # 使用之前定义的注册器注册 ID，并加入到总字典 neurons
        self.memory = 0.
        self.out = 0.
        self.inputs = {}
        self.memory_inputs = {}
        self.__neuron_id = neuron_register(self)
    def ID(self): # 返回自身ID值
        return self.__neuron_id
    def __str__(self) -> str: # 格式化可视化打印
        if self.inputs:
            n_str = '<RNN Neuron {1} at {0}, out = {2}, last out = {3}, inputs = ['.format(
                hex(id(self)), self.__neuron_id, self.out, self.memory)
            for k, w in self.inputs.items():
                n_str += '{0} : {1}, '.format(k, w)
            for k, w in self.memory_inputs.items():
                n_str += '{0} : {1}, '.format(k, w)
            n_str += '\b\b]>'
        else:
            n_str = '<RNN Neuron {1} at {0}, out = {2}, last_out = {3}>'.format(
                hex(id(self)), self.__neuron_id, self.out, self.memory)
        return n_str
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
    def __call__(self, mode = 'normal', activator_id = 1) -> float: # 计算输出
        if self.inputs:
            var = 0
            for x, w in self.inputs.items():
                var += w * neurons[x].out
            for x, w in self.memory_inputs.items():
                var += w * neurons[x].memory
            if mode == 'normal': # normal 将输出使用激活函数处理之后输出到 self.out
                self.out = activators[activator_id](var)
            elif mode == 'direct': # direct 将输出直接输出到 self.out
                self.out = var
            elif mode == 'derive': # derive 不赋值 self.out，只返回未处理的输出值
                return var
            return self.out
        else:
            pass
    def memorize(self): # 记录本轮输出作为“上一轮输出”
        self.memory = self.out
class Layer: # 神经网络中的一层结构
    __layer_id : int # 层 ID，作为私有属性不可更改
    neuron_dict : dict # 层内的神经元字典，可以根据 ID 查询
    __layer_type : str # 层类型，比如普通神经网络 BP 层，循环 RNN 层，径向基 RBF 层等，作为私有属性不可更改
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
        self.neuron_dict[neuron.ID()] = neuron
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
                for n in self.neurons():
                    w = [(key() if not args else key(**args)) for t in range(l)]
                    n.inputs = dict(zip(prevs, w))
            elif self.__layer_type == 'RBF':
                c_aux = []
                for n in prev_layer.neurons():
                    c_aux.append(n.out)
                m, M, p= min(c_aux), max(c_aux), len(c_aux)
                for n in self.neurons():
                    w = [(key() if not args else key(**args)) for t in range(l)]
                    c = [m + (M - m) * (.5 + j) / p for j in range(p)]
                    n.reset((sum((c_aux[j] - c[j]) ** 2 for j in range(p)) / p) ** .5)
                    n.inputs = dict(zip(prevs, w))
                    n.centers = dict(zip(prevs, c))
            elif self.__layer_type == 'RNN':
                for n in self.neurons():
                    w = [(key() if not args else key(**args)) for t in range(l)]
                    n.inputs = dict(zip(prevs, w))
        else:
            pass
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
                n(mode, activator_id)
        elif self.__layer_type == 'RBF':
            for n in self.neurons():
                n(mode)
        elif self.__layer_type == 'RNN':
            for n in self.neurons():
                n(mode, activator_id)
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
def spawn_RNN_layer(number_of_neurons : int) -> Layer:
    if number_of_neurons <= 0:
        return Null_layer
    else:
        l = Layer('RNN')
        for i in range(number_of_neurons):
            n = RNN_Neuron()
            l.neuron_dict[n.ID()] = n
        K = l.neuron_dict.keys()
        for n in l.neurons():
            n.memory_inputs = dict(zip(K,
                [random() for i in range(number_of_neurons)]))
        return l
def spawn_layer(number_of_neurons : int, layer_type : str = 'BP',
                key : callable = random, **args) -> Layer:
    if layer_type == 'BP': # 全连接层
        return spawn_BP_layer(number_of_neurons, key, **args)
    elif layer_type == 'RBF': # RBF 层
        return spawn_RBF_layer(number_of_neurons)
    elif layer_type == 'RNN': # RNN 层
        return spawn_RNN_layer(number_of_neurons)
    elif layer_type == 'IN': # Input 输入层
        l = Layer('IN')
        for i in range(number_of_neurons):
            n = Neuron()
            l.append(n)
        return l