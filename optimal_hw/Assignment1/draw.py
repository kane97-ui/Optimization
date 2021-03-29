import numpy as np
import matplotlib.pyplot as plt

#建立步长为0.01，即每隔0.01取一个点
step = 1
x = np.arange(-50,50,step)
y = np.arange(-50,50,step)
#也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份
x1=[-2,0]
x2=[1,0]
#将原始数据变成网格数据形式
X,Y = np.meshgrid(x,y)
#写入函数，z是大写
Z = X**4+2*(X-Y)*X**2+4*Y**2
# 设置打开画布大小,长10，宽6
plt.figure(figsize=(10,6))
# 填充颜色，f即filled
plt.contourf(X,Y,Z)
# 画等高线
plt.contour(X,Y,Z)
plt.scatter(x1,x2,marker='o',c='red')

plt.show()