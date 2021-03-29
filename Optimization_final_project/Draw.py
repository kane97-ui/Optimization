import matplotlib.pyplot as plt
import numpy as np
def plot_division(c1,c2,x_0,path,method):
    # plt.style.use('ggplot')
    plt.xlim(xmax = 50, xmin = 0)
    plt.ylim(ymax = 50, ymin = 0)
    plt.xlabel('a1')
    plt.ylabel('a2')
    plt.scatter(c1[0], c1[1], s=np.pi / 3, c='#DC143C')
    plt.scatter(c2[0], c2[1], s=np.pi / 3, c='#00CED1')
    x1 = np.arange(0, 50, 1)
    x2 = (-x_0[2] - x_0[0] * x1) / x_0[1]
    plt.plot(x1, x2,linewidth=1.5,label=method,c='green')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()
def gradient_plot(num_iteration,grad_norm,method):
    plt.style.use('ggplot')
    iteration = list(i for i in range(1, num_iteration + 1))
    grad_norm=np.log(np.asarray(grad_norm))
    # norm_grad = [np.log(i) for i in norm_grad]
    # plt.figure(1,figsize=(8,10))
    plt.plot(iteration,grad_norm,linewidth=1,label=method)
    plt.xlabel('number of iteration')
    plt.ylabel('$log||gradient||$')
    plt.legend()
    plt.tight_layout()

def acc_plot(accuracy,method):
    num_iteration=len(accuracy)
    plt.style.use('ggplot')
    iteration = list(i for i in range(1, num_iteration + 1))
    # norm_grad = [np.log(i) for i in norm_grad]
    # plt.figure(1,figsize=(8,10))
    plt.plot(iteration,accuracy,linewidth=1,label=method)
    plt.xlabel('number of iteration')
    plt.ylabel('$Accuracy$')
    plt.legend()
    plt.tight_layout()


 # plt.savefig(method+"_convergence",dpi=300)
# for i in range(4):
#     f_bt = open('svm_acc_bt_'+str(i)+'.txt')
#     acc_bt = f_bt.read().split(', ')
#     acc_bt_list=[]
#     for j in range(1,len(acc_bt)-1):
#         acc_bt_list.append(float(acc_bt[j]))
#
#     f_AGM=open('svm_acc_AGM_'+str(i)+'.txt')
#     acc_agm=f_AGM.read().split(', ')
#     acc_agm_list = []
#     for j in range(1,len(acc_agm)-1):
#         acc_agm_list.append(float(acc_agm[j]))
#
#     f_BFGS = open('svm_acc_BFGS_'+str(i)+'.txt')
#     acc_BFGS = f_BFGS.read().split(', ')
#     acc_BFGS_list = []
#     for j in range(1,len(acc_BFGS)-1):
#         acc_BFGS_list.append(float(acc_BFGS[j]))
#
#     acc_plot(acc_bt_list,'backtracking')
#     acc_plot(acc_agm_list, 'AGM')
#     acc_plot(acc_BFGS_list, 'BFGS')
#     plt.savefig('acc_'+str(i))
#     plt.show()
#     plt.close()
# plt.show()
#
# num=10
# num2=5
# grad1=[i for i in range(0,10)]
# grad2=[i for i in range(10,15)]
# gradient_plot(num,grad1,'s')
# gradient_plot(num2,grad2,'a')
# plt.show()
# a=[1,2,3]
# t=open('test.txt',mode='a+')
# t.write(np.str(a)+'33')