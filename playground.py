import numpy as np
from random import sample
import sys

inputs = np.array(
    [[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 1, 1], [4, 4, 4, 4], [3, 3, 3, 3]]
)

newinputs = inputs * 3
norms = np.linalg.norm(inputs, axis=1)
diffs = np.linalg.norm(newinputs - inputs, axis=1)


print(diffs / norms)
