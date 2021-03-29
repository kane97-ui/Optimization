from A3_3 import *
import numpy as np
import matplotlib.pyplot as plt
initial_points=[[-10,-2],[-2,2],[-10,2],[5,-2],[2,2],[10,2],[-5,-2],[-5,2],[-2,-2],[5,2]]
#建立步长为0.01，即每隔0.01取一个点
color_list=['blue','green','red','cyan','purple','yellow','orange','teal','coral','darkred']
method_list=['backtracking','exact_line_search','dimishing_step']
def contour_plot():
    step1 = 0.05
    step2=0.01125
    x1 = np.arange(-10.5,10.5,step1)
    x2 = np.arange(-2.045,3,step2)
    #也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份

    #将原始数据变成网格数据形式
    X1,X2 = np.meshgrid(x1,x2)
    #写入函数
    fx=(3+X1+((1-X2)*X2-2)*X2)**2+(3+X1+(X2-3)*X2)**2
    #设置打开画布大小,长10，宽6
    plt.figure(figsize=(10,6))
    #填充颜色，f即filled
    # plt.contourf(X1,X2,fx)
    #画等高线
    plt.contour(X1,X2,fx,15)
    # plt.show()
def path_plot(initial_point,color,method):
    # contour_plot()
    if method=='backtracking':
        xk_list_1, norm_grad_1, num_iteration_1 = backtracking(0.5, 0.1, 1e-5, initial_point)
    if method=='exact_line_search':
        xk_list_1, norm_grad_1, num_iteration_1 = exact_line_search(1e-5,initial_point)
    if method=='dimishing_step':
        xk_list_1, norm_grad_1, num_iteration_1 = dimishing_step(1e-5,initial_point)
    x1_list=[xk[0] for xk in xk_list_1]
    x2_list=[xk[1] for xk in xk_list_1]
    plt.plot(x1_list,x2_list,linewidth=1.5,color=color)
    plt.scatter(x1_list,x2_list,s=3)
    # plt.show()
def plot_different_method():
    for method in method_list:
        contour_plot()
        for i in range(10):
            path_plot(initial_points[i],color_list[i],method)
        # plt.show()
        plt.savefig(method + "_contour", dpi=300)
        plt.close()

plot_different_method()
