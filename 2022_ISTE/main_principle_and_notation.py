# -*- coding: utf-8 -*-
import numppy as np

A = np.array([[-1,1],[1,-0.5]])
B = np.linalg.inv(A)

print(A)
print(B)