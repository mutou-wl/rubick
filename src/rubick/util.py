#  返回的是 x 除以 y 的向上取整结果
def upDiv(x: int, y: int) -> int:
    return x // y + (0 if x % y == 0 else 1)
