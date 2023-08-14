print("Hello, world!")

from mip import Model, maximize
import numpy as np

m = Model()
x1 = m.add_var("x1", var_type="I")
x2 = m.add_var("x2", var_type="I")
x3 = m.add_var("x3", var_type="I")

m += x1 + x2 + x3 <= 10
m += x1 + 2 * x2 + 3 * x3 <= 15
m += x1 >= 0
m += x2 >= 0
m += x3 >= 0

m.objective = maximize(x1 + 4 * x2 + 3 * x3)

m.optimize()

print(x1.x, x2.x, x3.x)
