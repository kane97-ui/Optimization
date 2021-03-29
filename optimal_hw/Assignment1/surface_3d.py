import matplotlib.pyplot as plt
import numpy as np

#建立步长为0.01，即每隔0.01取一个点
ax3=plt.axes(projection='3d')
step = 0.1
x = np.arange(-50,50,step)
y = np.arange(-50,50,step)
#也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份
x1=[-2,0]
x2=[1,0]
X,Y = np.meshgrid(x,y)
Z = X**4+2*(X-Y)*X**2+4*Y**2
ax3.plot_surface(X,Y,Z,cmap='rainbow')
ax3.scatter(x1,x2,marker='*',c='red')
plt.show()