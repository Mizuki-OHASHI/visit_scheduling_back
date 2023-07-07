from mip import Model
import numpy as np
from random import randint

m = Model()
a = np.array([m.add_var("X", var_type="B"), 0])
print(a)
print(type(a[0]), type(a[1]))


print([(randint(0, 1), randint(0, 3)) for _ in range(10)])
