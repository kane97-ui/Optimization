from scipy import optimize as op
import numpy as np
c=np.array([130,230])
A_ub=np.array([[5,15],[4,4],[35,20],[1.5,2.5]])
B_ub=np.array([480,160,1190,84])
# A_eq=np.array([[1,1,1]])
# B_eq=np.array([7])
x1=(0,None)
x2=(0,None)
res=op.linprog(-c,A_ub,B_ub,bounds=(x1,x2))
print(res)
# print(68*200+150*52)
