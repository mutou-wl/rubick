import ast
from typing import List
from functools import reduce
import inspect

import numpy as np

from rubick.relation import Relation


class Iterator:
    def __init__(self, id, n):
        self.id = id # 迭代器id
        self.n = n  # 迭代域的总迭代器个数
        self.name: str ## 名称
        self.range: List[int] # 迭代器的范围

    def __add__(self, other): # 重载加法运算符，将当前迭代器与另一个对象相加
        return IteratorExpr(1, iterator=self.id, n=self.n) + IteratorExpr(other, n=self.n)

    def __radd__(self, other):# 重载右侧加法运算符，将另一个对象与当前迭代器相加
        return IteratorExpr(1, iterator=self.id, n=self.n) + IteratorExpr(other, n=self.n)

    def __sub__(self, other): # 重载减法运算符，将当前迭代器与另一个对象相减
        return IteratorExpr(1, iterator=self.id, n=self.n) - IteratorExpr(other, n=self.n)

    def __rsub__(self, other): # 重载右侧减法运算符，将另一个对象与当前迭代器相减
        return IteratorExpr(-1, iterator=self.id, n=self.n) + IteratorExpr(other, n=self.n)

    def __mul__(self, other): # 重载乘法运算符，将当前迭代器与一个整数相乘
        if isinstance(other, int):
            return IteratorExpr(other, iterator=self.id, n=self.n)
        raise TypeError("Unknown multiplier")

    def __rmul__(self, other): # 重载右侧乘法运算符，将一个整数与当前迭代器相乘
        if isinstance(other, int):
            return IteratorExpr(other, iterator=self.id, n=self.n)
        raise TypeError("Unknown multiplier")

    def setRange(self, lower, upper): # 设置迭代器的范围
        self.range = [lower, upper]

    def getSize(self): #  获取迭代器范围的大小
        return self.range[1] - self.range[0]


class IteratorExpr:
    def __init__(self, val, n, iterator=None, const=0):
        if iterator is not None:
            if isinstance(val, int):
                self.expr = np.zeros(n) # 创建一个长度为n的全零数组
                self.expr[iterator] = val # 在指定迭代器位置设置值
            else:
                raise TypeError("IteratorExpr: Unknown init")
        else:
            if isinstance(val, IteratorExpr):
                self.expr = val.expr
                self.n = val.n
                self.const = val.const
            elif isinstance(val, Iterator):
                self.expr = np.zeros(n)
                self.expr[val.id] = 1
                self.const = 0
            elif isinstance(val, np.ndarray):
                self.expr = val
            else:
                raise TypeError("IteratorExpr: Unknown init")

        self.n = n
        self.const = const

    def __add__(self, other): #接受与 int、Iterator、IteratorExpr 进行相加
        if isinstance(other, int):# int类型, const相加
            return IteratorExpr(self.expr, self.n, const=self.const + other)
        if isinstance(other, Iterator): # 与迭代器相加
            return other.__radd__(self)
        if isinstance(other, IteratorExpr): # 与迭代器表达式相加
            if self.n != other.n:
                raise RuntimeError("Incompatible iterator exprs")
            return IteratorExpr(self.expr + other.expr, self.n, const=self.const + other.const)
        raise TypeError("IteratorExpr: Unknown operand")

    def __radd__(self, other):
        if isinstance(other, int):
            return IteratorExpr(self.expr, self.n, const=self.const + other)
        raise TypeError("IteratorExpr: Unknown operand")

    def __sub__(self, other):
        if isinstance(other, int):
            return IteratorExpr(self.expr, self.n, const=self.const - other)
        if isinstance(other, Iterator):
            return other.__rsub__(self)
        if isinstance(other, IteratorExpr):
            if self.n != other.n:
                raise RuntimeError("Incompatible iterator exprs")
            return IteratorExpr(self.expr - other.expr, self.n, const=self.const + other.const)
        raise TypeError("IteratorExpr: Unknown operand")

    def __rsub__(self, other):
        if isinstance(other, int):
            return IteratorExpr(-self.expr, self.n, const=other - self.const)
        raise TypeError("IteratorExpr: Unknown operand")

    def __mul__(self, other): # 接受与 int、Iterator进行相乘,  但基本上只接受int
        if isinstance(other, int):
            return IteratorExpr(other * self.expr, self.n, const=other * self.const)
        if isinstance(other, Iterator):
            return other.__rmul__(self)
        raise TypeError("IteratorExpr: Unknown operand")

    def __rmul__(self, other):
        if isinstance(other, int):
            return IteratorExpr(other * self.expr, self.n, const=other * self.const)
        raise TypeError("IteratorExpr: Unknown operand")


class Tensor:
    """张量对象"""
    def __init__(self, id):
        self.id = id
        self.isOutput = False
        self.accFunc: Relation  # 关联的关系对象
        self.name: str

    def __getitem__(self, idx): #允许对象使用索引操作符 [] 来访问内部元素
        """重载的索引操作符: 获取张量的某个索引，返回一个 TensorIndex 对象"""
        if isinstance(idx, Iterator): # 迭代器转换为迭代表达式
            idx = IteratorExpr(1, n=idx.n, iterator=idx.id)
        if not isinstance(idx, IteratorExpr):
            raise TypeError("Tensor: Unknown index")
        return TensorIndex(self, idx)

    def setRange(self, *args): # (10,10) 就是两维, 每一维的取值范围从0开始
        """设置张量的范围"""
        self.range = args

    def domainStr(self):
        """返回张量定义域的字符串"""
        indices = [f"i{i+1}" for i in range(len(self.range))]
        return "{" + f"{self.name}[{','.join(indices)}]:{' and '.join([f'0<={i}<{r}' for (i,r) in zip(indices, self.range)])}" + "}"


class TensorExpr:
    """表示张量表达式的基类"""
    def __init__(self):
        self.sons = []

    def __add__(self, other):
        """重载加法运算符，返回一个 TensorAddExpr 对象"""
        if not isinstance(other, TensorExpr):
            raise TypeError("TensorExpr: Unknown operand")
        return TensorAddExpr(self, other)

    def __mul__(self, other):
        """重载乘法运算符，返回一个 TensorMulExpr 对象"""
        if not isinstance(other, TensorExpr):
            raise TypeError("TensorExpr: Unknown operand")
        return TensorMulExpr(self, other)

    def getAccFunc(self):

        return reduce(dictMerge, map(lambda x: x.getAccFunc(), self.sons))


class TensorIndex(TensorExpr):
    """张量的索引"""
    def __init__(self, tensor, idx, prev=None):
        super(TensorIndex, self).__init__()
        self.tensor = tensor
        self.idx = idx
        if prev is None:
            self.mat = [idx.expr.tolist()] # 索引矩阵
            self.const = [idx.const]  # 常量列表
        else:
            if not isinstance(prev, TensorIndex):
                raise TypeError("TensorIndex: Unknown prev")
            self.mat = prev.mat + [idx.expr.tolist()]
            self.const = prev.const + [idx.const]

    def __getitem__(self, idx):
        if isinstance(idx, Iterator):
            idx = IteratorExpr(1, n=idx.n, iterator=idx.id)
        if not isinstance(idx, IteratorExpr):
            raise TypeError("TensorIndex: Unknown index")
        return TensorIndex(self.tensor, idx, self)

    def getAccFunc(self):
        # return {self.tensor.name: {"AccFunc": self.mat, "Const": self.const}}
        return {self.tensor.name: {"accFunc": self.mat}}


class TensorBinaryExpr(TensorExpr):
    def __init__(self, son1, son2):
        super(TensorBinaryExpr, self).__init__()
        self.sons = [son1, son2]


class TensorAddExpr(TensorBinaryExpr):
    def __init__(self, son1, son2):
        super(TensorAddExpr, self).__init__(son1, son2)


class TensorMulExpr(TensorBinaryExpr):
    def __init__(self, son1, son2):
        super(TensorMulExpr, self).__init__(son1, son2)


class OpSpec:
    def __init__(self, name: str):
        self.name = name  # 操作名称
        self.numIter = 0  # 迭代器数量
        self.numTensor = 0 # 张量数量
        self.iterators: List[Iterator]  # 迭代器列表
        self.tensors: List[Tensor]  # 张量列表
        self.output: str  # 输出张量名称

    def genIterators(self, numIter): 
        """生成指定数量的迭代器"""
        self.numIter = numIter
        self.iterators = [Iterator(i, numIter) for i in range(numIter)] #创建numTter个迭代器
        context = inspect.stack()[1].code_context[0].strip()
        names = ast.parse(context).body[0].targets[0].elts
        for i in range(len(names)):
            self.iterators[i].name = names[i].id
        # 创建一个字典，将每个迭代器的名称映射到其索引
        self.name2iterator = {
            self.iterators[i].name: i
            for i in range(len(self.iterators))
        }
        return self.iterators

    def getIterator(self, nameOrId): 
        """用于通过名称或索引获取迭代器"""
        if isinstance(nameOrId, int):
            return self.iterators[nameOrId]
        else:
            return self.iterators[self.name2iterator[nameOrId]]

    def genTensors(self, numTensor):
        """生成指定数量的张量"""
        self.numTensor = numTensor
        self.tensors = [Tensor(i) for i in range(numTensor)]
        context = inspect.stack()[1].code_context[0].strip()
        names = ast.parse(context).body[0].targets[0].elts
        for i in range(len(names)):
            self.tensors[i].name = names[i].id
        self.name2tensor = {
            self.tensors[i].name: i
            for i in range(len(self.tensors))
        }
        return self.tensors

    def getTensor(self, nameOrId):
        """通过名称或索引获取张量"""
        if isinstance(nameOrId, int):
            return self.tensors[nameOrId]
        else:
            return self.tensors[self.name2tensor[nameOrId]]

    def makeIndices(self):
        """生成迭代器的名称列表"""
        return [i.name for i in self.iterators]

    def setExpr(self, leftValue, rightValue):
        """设置操作的表达式和输出"""
        self.expr = rightValue # TensorMulExpr
        self.output = leftValue # TensorIndex
        if isinstance(leftValue, TensorIndex):
            print(f"leftValue tensor: {leftValue.tensor.name}")
            print(f"leftValue idx: {leftValue.idx}")
            print(f"leftValue mat: {leftValue.mat}")
            print(f"leftValue const: {leftValue.const}")
        data = dictMerge(self.expr.getAccFunc(), leftValue.getAccFunc())
        leftValue.tensor.isOutput = True

        inIndices = self.makeIndices()

        for u, v in data.items():
            self.getTensor(u).accFunc = Relation(
                "S", inIndices, u, np.array(v['accFunc']))

        self.output = leftValue.tensor.name


def dictMerge(x, y):
    """合并两个字典"""
    z = {}
    for u, v in x.items():
        z[u] = v
    for u, v in y.items():
        z[u] = v
    return z


