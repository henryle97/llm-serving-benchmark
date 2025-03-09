import numpy as np


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    x = tuple((i, 1) for i in x)
    y = tuple(i for i in y)
    a, b = np.linalg.lstsq(x, y, rcond=None)[0]
    return a, b
