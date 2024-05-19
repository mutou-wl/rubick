import time
from rubick.dataflowDSE import DataflowDSE
from rubick.relation import *
from rubick.ir import *
from rubick.interface import *
from rubick.perfModel import PerfModel




if __name__ == "__main__":
    #创建两个迭代器对象
    iterator1 = Iterator(id=0, n=3)
    iterator2 = Iterator(id=1, n=3)
    # 设置迭代器的范围
    iterator1.setRange(0, 10)
    iterator2.setRange(5, 15)
    # 显示迭代器的大小
    # print(f"Iterator 1 size: {iterator1.getSize()}")
    # print(f"Iterator 2 size: {iterator2.getSize()}")
    # 迭代器的加法运算
    expr1 = iterator1 + iterator2  #返回的是IteratorExpr对象
    # print(f"Expr1 after addition: {expr1.expr}, Const: {expr1.const}")
    # 迭代器的减法运算
    expr2 = iterator1 - iterator2
    # print(f"Expr2 after subtraction: {expr2.expr}, Const: {expr2.const}")
    # 迭代器与整数的乘法运算
    expr3 = iterator1 * 5
    # print(f"Expr3 after multiplication: {expr3.expr}, Const: {expr3.const}")
    # 迭代器表达式与整数的加法运算
    expr4 = expr3 + 10
    # print(f"Expr4 after addition with constant: {expr4.expr}, Const: {expr4.const}")
    # 迭代器表达式与整数的减法运算
    expr5 = expr4 - 5
    # print(f"Expr5 after subtraction with constant: {expr5.expr}, Const: {expr5.const}")



# Tensor 类
    # 实例化 Tensor 对象
    tensor1 = Tensor(id=0)
    tensor1.name = "A"
    tensor1.setRange(10, 10) #两维
    tensor2 = Tensor(id=1)
    tensor2.name = "B"
    tensor2.setRange(10, 10)
    print(f"Tensor1:   name:{tensor1.name} id:{tensor1.id} 定义域: {tensor1.domainStr()}")
    iterator = Iterator(id=0, n=3)
    iterator.name = "i"
    iterator.setRange(0, 10)
    iteratorExpr = IteratorExpr(iterator, 3) #迭代表达式
    print(f"IteratorExpr: expr:{iteratorExpr.expr}, Const: {iteratorExpr.const}")





    # 初始化一个OpSpec对象
    # op = OpSpec("example_op")
    # # 生成3个迭代器
    # i, j, k = op.genIterators(3)
    # A, B, C = op.genTensors(3)
    # i.setRange(0, 16)
    # j.setRange(0, 16)
    # k.setRange(0, 16)
    # # Tensor
    # A.setRange(16, 16)
    # B.setRange(16, 16)
    # C.setRange(16, 16)
    # op.setExpr(C[i][j], A[i][k] * B[k][j])

    # iterators = [i, j, k]
    # print("生成的迭代器:")
    # for iterator in iterators:
    #     print(f"迭代器名称: {iterator.name}, 索引: {iterator.id}, n:{iterator.n}, 范围: {iterator.range}, 大小: {iterator.getSize()}")
    # itername = op.makeIndices()
    # print("生成的迭代器名称: ", itername)
