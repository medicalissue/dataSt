import numpy as np
Aa = np.array([1])
Ab = Aa
Aa[0] = 0
print(Ab[0])
La = [1]
Lb = La
La[0] = 0
print(Lb[0])
La.append(3)
print(Lb[1])