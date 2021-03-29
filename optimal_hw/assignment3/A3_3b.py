from A3_3 import *
import matplotlib.pyplot as plt
xk_list_1,norm_grad_1,num_iteration_1=backtracking(0.5,0.1,1e-5,[0,0])
xk_list_2,norm_grad_2,num_iteration_2=exact_line_search(1e-5,[0,0])
xk_list_3,norm_grad_3,num_iteration_3=dimishing_step(1e-5,[0,0])
xk_list=[xk_list_1,xk_list_2,xk_list_3]
norm_grad_list=[norm_grad_1,norm_grad_2,norm_grad_3]
num_iteration_list=[num_iteration_1,num_iteration_2,num_iteration_3]
method_list=['backtracking','exact_line_search','dimishing_step']
color_list=['blue','red','green']
def gradient_plot(num_iteration,norm_grad,method,color):
    iteration = list(i for i in range(1, num_iteration + 1))
    norm_grad = [log(i) for i in norm_grad]
    # plt.figure(1,figsize=(8,10))
    plt.plot(iteration,norm_grad,label=method,color=color,linewidth=2)
    plt.xlabel('number of iteration')
    plt.ylabel('$log||gradient||$')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(method+"_gradient",dpi=300)
    plt.close()
def norm2(xk,x_star):
    return (xk[0]-x_star[0])**2+(xk[1]-x_star[1])**2
def xk_plot(num_iteration,xk_li,method,color):
    iteration = list(i for i in range(1, num_iteration+2))
    x_star=(-1,1)
    xk = [log(norm2(j,x_star)**0.5) for j in xk_li]
    plt.plot(iteration,xk,label=method,color=color,linewidth=2)
    plt.xlabel('number of iteration')
    plt.ylabel('$log||(x^k-x^*)||$')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(method+"_xk",dpi=300)
    plt.close()
for i in range(3):
    num_iteration=num_iteration_list[i]
    norm_grad=norm_grad_list[i]
    method=method_list[i]
    color=color_list[i]
    gradient_plot(num_iteration,norm_grad,method,color)
for i in range(3):
    num_iteration=num_iteration_list[i]
    xk_li=xk_list[i]
    method=method_list[i]
    color=color_list[i]
    xk_plot(num_iteration,xk_li,method,color)