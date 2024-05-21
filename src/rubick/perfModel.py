from multiprocessing.util import MAXFD
from typing import Iterator, List
import json

import numpy as np
from rubick.dataflowDSE import DataflowDSE

from rubick.relation import *
from rubick.ir import *


class PerfModel:
    """性能模型类"""
    def __init__(self, arraySpec):
        self.arraySpec = arraySpec

    def __call__(
        self,
        ops: List[OpSpec],
        maxBufDim: int,     # Buf的维度  
        bufSizeLimit: int,  # Buf的大小限制  65536
        bwCapcity: int,     # 位宽的容量限制  2.56
        exactReuse: bool,   # 默认为False, 有bug 不用管
        outFile: str        # 输出文件
    ):
        """进行数据流分析，并将结果输出到指定文件"""

        dse = DataflowDSE(self.arraySpec)  #传入PE阵列的信息,  生成DSE空间, 用来迭代
        with open(outFile, "w") as fout:   #开始搜索 
            fout.write("[")
            first = True
            for accEntries, dataflowGens in dse(ops, None, exactReuse):
                opDataflow = [[d for d in dataflowGen]
                              for dataflowGen in dataflowGens]
                if not reduce(lambda a, b: a and b, map(lambda x: len(x) > 0, opDataflow)): 
                    continue #检查每个生成器是否为空，如果为空则跳过。
                curAccEntries = {"accEntries": [
                    str(e) for e in accEntries], "op": {}}

                ok = True

                for op, opD in zip(ops, opDataflow):
                    cur = []
                    for d in opD:
                        lef = 1
                        rig = min(maxBufDim, d.timeDims)
                        while lef != rig: # 二分法: 尽可能的占满 buf
                            mid = (lef + rig + 1) // 2
                            bufSize = d.bufferSize(mid)
                            if bufSize > bufSizeLimit:
                                rig = mid - 1
                            else:
                                lef = mid
                        bufDim = lef
                        # bufDim = 1
                        bufSize = d.bufferSize(bufDim)
                        if bufSize > bufSizeLimit:
                            continue
                        bwReq = bufSize / d.tileTime(bufDim)
                        latency = d.peakLatency() * max(1.0, bwReq / bwCapcity)
                        # latency = d.peakLatency()

                        cur.append({
                            "dataLayouts": [str(l) for l in d.dataLayouts],
                            "bufDim": bufDim,
                            "bufSize": bufSize,
                            "bwRequirement": bwReq,
                            "latency": latency,
                            "spaceRange": d.spaceRange
                        })
                    if len(cur):
                        curAccEntries["op"][op.name] = cur
                    else:
                        ok = False
                        break

                if ok:
                    if not first:
                        fout.write(",\n")
                    else:
                        first = False
                    fout.write(json.dumps(curAccEntries, indent=2))
            fout.write("]\n")
