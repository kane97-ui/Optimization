import scipy.io as scio
from scipy.sparse import hstack
import numpy as np
import logistic_regression as LR
import SVM
import Data_preparation as load
import argparse
import Draw
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('-l',type=float,help='lambda',default=0.0001)
parser.add_argument('-d',type=float,help='delta',default=0.1)
parser.add_argument('-m', type=int,help='size of memory',default=5)

parser.add_argument('-dtrain_s',type=str,help='training data path',default='datasets/breast-cancer/breast-cancer_train.mat')
parser.add_argument('-label1_s',type=str,help='path of label',default='datasets/breast-cancer/breast-cancer_train_label.mat')

parser.add_argument('-dtrain_m',type=str,help='training data path',default='datasets/mushrooms/mushrooms_train.mat')
parser.add_argument('-label1_m',type=str,help='path of label',default='datasets/mushrooms/mushrooms_train_label.mat')

parser.add_argument('-dtrain_l',type=str,help='training data path',default='datasets/rcv1/rcv1_train.mat')
parser.add_argument('-label1_l',type=str,help='path of label',default='datasets/rcv1/rcv1_train_label.mat')
parser.add_argument('-dtest_l',type=str,help='training data path',default='datasets/rcv1/rcv1_test.mat')
parser.add_argument('-label2_l',type=str,help='path of label',default='datasets/rcv1/rcv1_test_label.mat')

#
args = parser.parse_args()
#
#
# train_data=scio.loadmat(args.dtrain_l)
# train_label=scio.loadmat(args.label1_l)
#
#
# test_data=scio.loadmat(args.dtest_l)
# test_label=scio.loadmat(args.label2_l)
#
# a=train_data['A']
# b=train_label['b']
#
# a1=test_data['A']
# b1=test_label['b']
# # a=a[:int(0.7*np.shape(a)[0])]
# # a1=a[int(0.7*np.shape(a)[0]):]
# # b = b[:int(0.7 * np.shape(b)[0])]
# # b1 = b[int(0.7 * np.shape(b)[0]):]
#
# # m,n=np.shape(a)
# m1,n1=np.shape(a1)
# # a=hstack((a, np.ones((m,1))))
# a1=hstack((a1, np.ones((m1,1))))
# m,n=np.shape(a)
# x=np.random.random((n+1,1))
# # x_0,norm,iteration=lr.L_BFGS(a,b,x,args.m,args.l,True,a1,b1)
# batch_size=[100,500,800,1000]
# x_0,norm,iteration,acc,duration,duration_iteration=LR.sgd_method(a,b,x,20000,args.l,True,a1,b1)
# f = open('sgd_acc_4'+'.txt', mode='a+')
# f.write(np.str(acc)+str(duration)+' '+str(duration_iteration)+np.str(norm))



# x_bt,norm_bt,iteration_bt,acc_bt,duration_bt,diter_bt=SVM.BackTracking(a,b,x,args.l,args.d,False,a1,b1)
data=[load.c1,load.c2,load.c3,load.c4]
C1=[load.c1_1,load.c2_1,load.c3_1,load.c4_1]
C2=[load.c1_2,load.c2_2,load.c3_2,load.c4_2]
Label=[load.label1,load.label2,load.label3,load.label4]
def plot_convergence_synthetic(svm,lr):
    for i in range(4):
        a = data[i]
        b = Label[i]
        n, m = np.shape(a)
        a = np.vstack((a, np.ones((1, m))))
        b = b.reshape(-1, 1)
        x = np.zeros((3, 1))
        a1 = a
        b1 = b
        if svm:
            x_bt, norm_bt,_, acc_bt, duration_bt, diteration_bt = SVM.BackTracking(a, b, x, args.l, args.d, False, a1, b1)
            f_bt = open('svm_acc_bt_'+str(i)+'.txt', mode='a+')
            f_bt.write(np.str(acc_bt)+' '+str(duration_bt)+' '+str(diteration_bt))

            x_AGM, norm_AGM, _, acc_AGM, duration_AGM, diteration_AGM  = SVM.AGM(a, b, x, args.l, args.d, False, a1, b1)
            f_AGM = open('svm_acc_AGM_'+str(i)+'.txt', mode='a+')
            f_AGM.write(np.str(acc_AGM)+' '+str(duration_AGM)+' '+str(diteration_AGM))

            x_BFGS, norm_BFGS, _, acc_BFGS, duration_BFGS, diteration_BFGS = SVM.G_BFGS(a, b, x, args.l, args.d, False, a1, b1)
            f_BFGS = open('svm_acc_BFGS_' + str(i) + '.txt', mode='a+')
            f_BFGS.write(np.str(acc_BFGS) + ' ' + str(duration_BFGS) + ' ' + str(diteration_BFGS))

            Draw.gradient_plot(len(norm_bt),norm_bt,'backtracking')
            Draw.gradient_plot(len(norm_AGM), norm_AGM, 'AGM')
            Draw.gradient_plot(len(norm_BFGS), norm_BFGS, 'BFGS')
            plt.savefig('SVM_convergence_'+str(i))
            # plt.show()
            plt.close()
        if lr:
            x_AGM, norm_AGM, _, acc_AGM, duration_AGM, diteration_AGM = LR.AGM(a, b, x, args.l, False, a1,b1)
            f_AGM = open('lr_acc_AGM_' + str(i) + '.txt', mode='a+')
            f_AGM.write(np.str(acc_AGM) + ' ' + str(duration_AGM) + ' ' + str(diteration_AGM))

            x_L_BFGS, norm_L_BFGS, _, acc_L_BFGS, duration_L_BFGS, diteration_L_BFGS = LR.L_BFGS(a, b, x, args.m,args.l, False, a1, b1)
            f_L_BFGS = open('lr_acc_LBFGS_' + str(i) + '.txt', mode='a+')
            f_L_BFGS.write(np.str(acc_L_BFGS) + ' ' + str(duration_L_BFGS) + ' ' + str(diteration_L_BFGS))

            Draw.gradient_plot(len(norm_AGM),norm_AGM,'AGM')
            Draw.gradient_plot(len(norm_L_BFGS), norm_L_BFGS, 'L_BFGS')
            plt.savefig('lr_convergence_'+str(i))
            # plt.show()
            plt.close()

def plot_convergence_RealData(size,train,tr_label,test='',te_label=''):
    if size != 'l':
        train_data = scio.loadmat(train)
        train_label = scio.loadmat(tr_label)
        a = train_data['A']
        b = train_label['b']
        a=a[:int(0.7*np.shape(a)[0])]
        a1=a[int(0.7*np.shape(a)[0]):]
        b = b[:int(0.7 * np.shape(b)[0])]
        b1 = b[int(0.7 * np.shape(b)[0]):]
        m, n = np.shape(a)
        m1,n1=np.shape(a1)
        a = hstack((a, np.ones((m, 1))))
        a1 = hstack((a1, np.ones((m1, 1))))
        m, n = np.shape(a)
        x = np.zeros((n, 1))
    else:
        train_data = scio.loadmat(train)
        train_label = scio.loadmat(tr_label)
        test_data = scio.loadmat(test)
        test_label = scio.loadmat(te_label)

        a = train_data['A']
        b = train_label['b']

        a1 = test_data['A']
        b1 = test_label['b']

        m, n = np.shape(a)
        m1, n1 = np.shape(a1)
        a = hstack((a, np.ones((m, 1))))
        a1 = hstack((a1, np.ones((m1, 1))))
        m, n = np.shape(a)
        x = np.zeros((n, 1))

    x_LBFGS, norm_LBFGS, _, acc_LBFGS, duration_LBFGS, diteration_LBFGS = LR.L_BFGS(a, b, x, args.m,args.l, True, a1, b1)
    f_LBFGS = open('LR_'+size+'.txt', mode='a+')
    f_LBFGS.write(np.str(acc_LBFGS) + ' ' + str(duration_LBFGS) + ' ' + str(diteration_LBFGS))

    x_BFGS, norm_BFGS, _, acc_BFGS, duration_BFGS, diteration_BFGS = SVM.L_BFGS(a, b, x, args.m,args.l, args.d, True, a1, b1)
    f_BFGS = open('svm_'+size+'.txt', mode='a+')
    f_BFGS.write(np.str(acc_BFGS) + ' ' + str(duration_BFGS) + ' ' + str(diteration_BFGS))

    Draw.gradient_plot(len(norm_BFGS), norm_BFGS, 'SVM')
    Draw.gradient_plot(len(norm_LBFGS), norm_LBFGS, 'LR')
    plt.savefig('SVM_VS_LR_LBFGS_'+size)
    plt.show()
    plt.close()


def plot_accuracy(svm,lr):
    if svm:
        for i in range(4):
            f_bt = open('svm_acc_bt_'+str(i)+'.txt')
            acc_bt = f_bt.read().split(', ')
            acc_bt_list=[]
            for j in range(1,len(acc_bt)-1):
                acc_bt_list.append(float(acc_bt[j]))

            f_AGM=open('svm_acc_AGM_'+str(i)+'.txt')
            acc_agm=f_AGM.read().split(', ')
            acc_agm_list = []
            for j in range(1,len(acc_agm)-1):
                acc_agm_list.append(float(acc_agm[j]))

            f_BFGS = open('svm_acc_BFGS_'+str(i)+'.txt')
            acc_BFGS = f_BFGS.read().split(', ')
            acc_BFGS_list = []
            for j in range(1,len(acc_BFGS)-1):
                acc_BFGS_list.append(float(acc_BFGS[j]))

            Draw.acc_plot(acc_bt_list,'backtracking')
            Draw.acc_plot(acc_agm_list, 'AGM')
            Draw.acc_plot(acc_BFGS_list, 'BFGS')
            plt.savefig('acc_'+str(i))
            plt.show()
            plt.close()
    if lr:
        for i in range(4):
            f_AGM = open('lr_acc_AGM_' + str(i) + '.txt')
            acc_agm = f_AGM.read().split(', ')
            acc_agm_list = []
            for j in range(1, len(acc_agm) - 1):
                acc_agm_list.append(float(acc_agm[j]))

            f_LBFGS = open('lr_acc_LBFGS_' + str(i) + '.txt')
            acc_LBFGS = f_LBFGS.read().split(', ')
            acc_LBFGS_list = []
            for j in range(1, len(acc_LBFGS) - 1):
                acc_LBFGS_list.append(float(acc_LBFGS[j]))

            Draw.acc_plot(acc_agm_list, 'AGM')
            Draw.acc_plot(acc_LBFGS_list, 'LBFGS')
            plt.savefig('lr_acc_' + str(i))
            plt.show()
            plt.close()
def plot_realdata_accuracy(file,label):
    f1= open(file)
    acc = f1.read().split(', ')
    acc_list = []

    for j in range(1, len(acc) - 1):
        acc_list.append(float(acc[j]))

    Draw.acc_plot(acc_list, label)

def plot_sgd_norm(file,label):
    f1= open(file)
    norm = f1.read().split(', ')
    norm_list = []
    for j in range(len(norm)):
        norm_list.append(float(norm[j]))
    Draw.gradient_plot(len(norm_list), norm_list, label)
# plot_sgd_norm('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/norm_1.txt','batch_size=1024')
# plot_sgd_norm('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/norm_2.txt','batch_size=2048')
# plot_sgd_norm('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/norm_3.txt','batch_size=4096')
# plot_sgd_norm('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/norm_4.txt','batch_size=10000')
# plt.savefig('sgd_norm')
# plt.show()
# plot_realdata_accuracy('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/sgd_acc_1.txt','batch_size=1024')
# plot_realdata_accuracy('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/sgd_acc_2.txt','batch_size=2048')
# plot_realdata_accuracy('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/sgd_acc_3.txt','batch_size=4096')
# plot_realdata_accuracy('/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/sgd_acc_4.txt','batch_size=10000')
# plt.savefig('sgd_batch')
# plt.show()
# plot_realdata_accuracy('l','/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/svm_l.txt','/Users/kane/Documents/optimization/final_project/part2/svm_vs_lr/LR_l.txt')



def plot_division(svm,lr):
    for i in range(4):
        a = data[i]
        c1 = C1[i]
        c2 = C2[i]
        b = Label[i]
        n, m = np.shape(a)
        a = np.vstack((a, np.ones((1, m))))
        b = b.reshape(-1, 1)
        x = np.zeros((3, 1))
        a1 = a
        b1 = b
        if svm:
            path_BT = 'svm_BT_a' + str(i)
            path_AGM = 'svm_AGM_a'+str(i)
            path_BFGS= 'svm_BFGS_a' + str(i)
            x_bt, norm_bt,_, acc_bt, duration_bt, diteration_bt = SVM.BackTracking(a, b, x, args.l, args.d, False, a1, b1)

            x_AGM, norm_AGM, _, acc_AGM, duration_AGM, diteration_AGM  = SVM.AGM(a, b, x, args.l, args.d, False, a1, b1)

            x_BFGS, norm_BFGS, _, acc_BFGS, duration_BFGS, diteration_BFGS = SVM.G_BFGS(a, b, x, args.l, args.d, False, a1, b1)

            Draw.plot_division(c1,c2,x_AGM,path_BT,'svm_BT')
            Draw.plot_division(c1, c2, x_AGM, path_AGM, 'svm_AGM')
            Draw.plot_division(c1, c2, x_BFGS, path_BFGS, 'svm_BFGS')
        if lr:
            path_AGM = 'lr_AGM_a'+str(i)
            path_LBFGS= 'lr_LBFGS_a' + str(i)
            x_AGM, norm_AGM, _, acc_AGM, duration_AGM, diteration_AGM = LR.AGM(a, b, x, args.l, False, a1,b1)
            x_L_BFGS, norm_L_BFGS, _, acc_L_BFGS, duration_L_BFGS, diteration_L_BFGS = LR.L_BFGS(a, b, x, args.m,args.l, False, a1, b1)

            Draw.plot_division(c1, c2, x_AGM, path_AGM, 'AGM')
            Draw.plot_division(c1, c2, x_L_BFGS, path_LBFGS, 'LBFGS')



def adjustment_m(size,lamda,train,tr_label,test='',te_label=''):
    if size != 'l':
        train_data = scio.loadmat(train)
        train_label = scio.loadmat(tr_label)
        a = train_data['A']
        b = train_label['b']
        a=a[:int(0.7*np.shape(a)[0])]
        a1=a[int(0.7*np.shape(a)[0]):]
        b = b[:int(0.7 * np.shape(b)[0])]
        b1 = b[int(0.7 * np.shape(b)[0]):]
        m, n = np.shape(a)
        m1,n1=np.shape(a1)
        a = hstack((a, np.ones((m, 1))))
        a1 = hstack((a1, np.ones((m1, 1))))
        m, n = np.shape(a)
        x = np.zeros((n, 1))
    else:
        train_data = scio.loadmat(train)
        train_label = scio.loadmat(tr_label)
        test_data = scio.loadmat(test)
        test_label = scio.loadmat(te_label)

        a = train_data['A']
        b = train_label['b']

        a1 = test_data['A']
        b1 = test_label['b']

        m, n = np.shape(a)
        m1, n1 = np.shape(a1)
        a = hstack((a, np.ones((m, 1))))
        a1 = hstack((a1, np.ones((m1, 1))))
        m, n = np.shape(a)
        x = np.zeros((n, 1))
    for i in range(len(lamda)):
        x_LBFGS, norm_LBFGS, _, acc_LBFGS, duration_LBFGS, diteration_LBFGS = LR.L_BFGS(a, b, x, 5, lamda[i], True, a1,
                                                                                    b1)
        f_LBFGS = open('parameter_lamda_' + size+'_'+str(lamda[i]) + '.txt', mode='a+')
        f_LBFGS.write(np.str(acc_LBFGS) + ' ' + str(duration_LBFGS) + ' ' + str(diteration_LBFGS))
        # Draw.gradient_plot(len(norm_LBFGS), norm_LBFGS, 'm='+str(lamda[i]))
    # plt.savefig('parameter_'+size)
    # plt.show()
    # plt.close()
plot_convergence_RealData('l', args.dtrain_l, args.label1_l, args.dtest_l, args.label2_l)
# lamda=[0.0001,0.001,0.1,0.3]
# adjustment_m('s',lamda,args.dtrain_s,args.label1_s)
# adjustment_m('m',lamda,args.dtrain_m,args.label1_m)
# adjustment_m('l',lamda,args.dtrain_l,args.label1_l,args.dtest_l,args.label2_l)

# plot_convergence(0,1)
# plot_convergence_RealData('m',args.dtrain_m,args.label1_m)
# plot_realdata_accuracy('LR_acc_LBFGS_realData_s.txt','LR_LBFGS')
# plot_realdata_accuracy('svm_acc_BFGS_realData_mushrooms.txt','SVM_BFGS')
# plt.savefig('svm_vs_lr_s')
# plt.show()
# plot_division(0,1)
    # Draw.plot_division(c1,c2,x_AGM,path_AGM,'AGM')
    # Draw.plot_division(c1, c2, x_BFGS, path_BFGS, 'BFGS')
# x_agm,norm_agm,iteration_agm,acc_agm,duration_agm,diter_agm=SVM.AGM(a,b,x,args.l,args.d,False,a1,b1)
# x_bfgs,norm_bfgs,iteration_bfgs,acc_bfgs,duration_bfgs,diter_bfgs=SVM.G_BFGS(a,b,x,args.l,args.d,False,a1,b1)

# train_data = scio.loadmat(args.dtrain_s)
# train_label = scio.loadmat(args.label1_s)
# b = train_label['b']
# a = train_data['A']
# a=a[:int(0.7*np.shape(a)[0])]
# b=b[:int(0.7*np.shape(b)[0])]
# m, n = np.shape(a)
# a = hstack((a, np.ones((m, 1))))
# # print(np.shape(a)[0])
#
# print(np.shape(a))

