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
    # A = Tensor(id=0)
    # A.name = "A"
    # A.setRange(10, 10) #两维
    # B = Tensor(id=1)
    # B.name = "B"
    # B.setRange(10, 10)
    # print(f"Tensor1: name:{A.name} id:{A.id} 定义域: {A.domainStr()}")
    # print(f"Tensor2: name:{B.name} id:{B.id} 定义域: {B.domainStr()}")
    # i = Iterator(id=0, n=3)
    # i.name = "i"
    # i.setRange(0, 10)
    # j = Iterator(id=1, n=3)
    # j.name = "j"
    # j.setRange(0, 10)
    # k = Iterator(id=2, n=3)
    # k.name = "k"
    # k.setRange(0, 10)
    # iteratorExpr = IteratorExpr(k, 3) #迭代表达式
    # print(f"IteratorExpr: expr:{iteratorExpr.expr}, Const: {iteratorExpr.const}")
    # tensorIndexA = A[i][k]
    # tensorIndexB = B[k][j]
    # print(f"TensorIndex: tensor[{tensorIndexA.tensor.name}] mat:{tensorIndexA.mat}, const:{tensorIndexA.const}")
    # print(f"TensorIndex: tensor[{tensorIndexB.tensor.name}] mat:{tensorIndexB.mat}, const:{tensorIndexB.const}")
    # # 打印 getAccFunc 返回值
    # print("tensorIndexA getAccFunc:", tensorIndexA.getAccFunc())
    # print("tensorIndexB getAccFunc:", tensorIndexB.getAccFunc())
    # tensor_mul_expr = tensorIndexA+tensorIndexB
    # print("TensorMulExpr getAccFunc:", tensor_mul_expr.getAccFunc())

    






    # 初始化一个OpSpec对象
    opSpec = OpSpec("example_op")
    # 生成3个迭代器
    i, j, k = opSpec.genIterators(3)
    A, B, C = opSpec.genTensors(3)
    # 设计迭代器和张量范围
    i.setRange(0, 16)
    j.setRange(0, 16)
    k.setRange(0, 16)
    A.setRange(16, 16)
    B.setRange(16, 16)
    C.setRange(16, 16)
    # 设计张量表达式
    opSpec.setExpr(C[i][j], A[i][k] * B[k][j])
    # 加载PE的设置
    arraySpec = ArraySpec("data/2D_entry_test.json") 
    # 建立数据流DSE对象
    dse = DataflowDSE(arraySpec) 
    opSpecs =[opSpec]
    k = 0
    for accEntries, dataflowGens in dse(opSpecs, None, False):
        print(f"迭代: {k}")

        print(accEntries[0].name, accEntries[0].relation)
        print(accEntries[1].name, accEntries[1].relation)
        print(accEntries[2].name, accEntries[2].relation)
    
        # systolic = set()
        # multicast = set()
        # stationary = False
        # for v in accEntries[2].vecs:
        #     if v[accEntries[2].arraySpec.spaceDims] != 0: # 方向向量时间维度第一个为0 就为multicast
        #         if np.sum(v != 0) > 1:
        #             systolic.add(tuple(v.tolist()))
        #         else:
        #             stationary = True
        #     else:
        #         multicast.add(tuple(v.tolist()))
        # print("Systolic 内容:")
        # for item in systolic:
        #     print(item)
        # print("multicast 内容:")
        # for item in multicast:
        #     print(item)
        k += 1
        if k==2:
            break

    # iterators = [i, j, k]
    # print("生成的迭代器:")
    # for iterator in iterators:
    #     print(f"迭代器名称: {iterator.name}, 索引: {iterator.id}, n:{iterator.n}, 范围: {iterator.range}, 大小: {iterator.getSize()}")
    # itername = op.makeIndices()
    # print("生成的迭代器名称: ", itername)
