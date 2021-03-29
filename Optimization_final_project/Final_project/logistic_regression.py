import numpy as np
import Data_preparation as load
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import Draw
import time

def norm_func(x):
    return np.sum(np.square(x))
def norm_s(x):
    return np.sqrt(np.sum(np.square(x)))
def obj_function(a,b,x,Lambda,sparse=False):
    if sparse:
        linear=a*x
    else:
        linear=np.dot(a.T,x)
    active=np.exp(np.multiply(-b,linear))+1
    sigmoid=np.log(active)
    part1=np.average(sigmoid)
    part2=Lambda/2*norm_func(x[:-1])
    return part1+part2

def Gradient(a,b,x,Lambda,sparse=False):
    if sparse:
        linear = a * x
    else:
        linear = np.dot(a.T, x)
    m,n=np.shape(a)
    active = np.exp(np.multiply(-b, linear)) + 1
    if sparse:
        grad_x = -a.T*np.multiply((np.true_divide(active - 1, active)), b) /m + Lambda * np.vstack(
            (x[:-1], np.asarray([[0]])))
    else:
        grad_x=np.dot(-a,np.multiply((np.true_divide(active-1,active)),b))/n+Lambda*np.vstack((x[:-1],np.asarray([[0]])))
    return grad_x

def S(x_1,x_0):
    return x_1-x_0

def Y(x_1,x_0,a,b,Lambda,sparse):
    return Gradient(a,b,x_1,Lambda,sparse)-Gradient(a,b,x_0,Lambda,sparse)

def P(s,y):
    return 1/np.dot(s.T,y)

def H_0(s,y):
    m,n=np.shape(s)
    return np.dot(s.T,y)/norm_func(y)*np.eye(m)

def direction(x,s,y,a,b,Lambda,sparse):
    return np.dot(H_0(s,y),Gradient(a,b,x,Lambda,sparse))

def backtracking(a,b,Lambda,sigma,gama,direction,x,sparse):
        alpha = 1
        while obj_function(a,b,x+alpha*direction,Lambda,sparse)-obj_function(a,b,x,Lambda,sparse)>gama*alpha*(np.dot(Gradient(a,b,x,Lambda,sparse).T,direction)):
            alpha=alpha*sigma
        return alpha

def store(li,value,limitation):
    if len(li)<limitation:
        li.append(value)
    else:
        li.append(value)
        li=li[1:]
    return li

def L_BFGS(a,b,x,m,Lambda,sparse,a1,b1):
    if sparse:
        num, n = np.shape(a)
    else:
        n, num = np.shape(a)
    x_0=x
    d = -Gradient(a,b,x_0,Lambda,sparse)
    s=[]
    y=[]
    norm_list=[]
    iter=1
    duration=0
    acc_list=[]
    while iter<100:
        start_time=time.time()
        alpha=backtracking(a,b,Lambda,0.5, 0.1, d, x_0,sparse)
        # print(alpha)
        x_1 = x_0 + alpha * d
        acc=accuracy(x_1,a1,b1,sparse)
        acc_list.append(acc)
        norm=np.sqrt(norm_func(Gradient(a,b,x_1,Lambda,sparse)))
        norm_list.append(norm)
        print('iteration:{} loss:{} norm:{} accuracy:{}'.format(iter, obj_function(a,b,x_1,Lambda,sparse), norm,acc))
        if norm<= 10 ** (-5):
            break
        s0 = S(x_1, x_0)
        y0 = Y(x_1,x_0,a,b,Lambda,sparse)
        if np.dot(s0.T,y0)<=10**(-14):
            h=np.eye(n)
        else:
            h=H_0(s0,y0)
        q=Gradient(a,b,x_1,Lambda,sparse)
        s=store(s,s0,m)
        y=store(y,y0,m)
        for i in range(len(s)-1,-1,-1):
            if np.dot(s[i].T,y[i])<=10**(-14):
                continue
            alpha=P(s[i],y[i])*np.dot(s[i].T,q)
            q-=alpha*y[i]
        r=np.dot(h,q)
        for i in range(len(s)):
            if np.dot(s[i].T,y[i])<=10**(-14):
                continue
            beta=P(s[i],y[i])*np.dot(y[i].T,r)
            r+=(P(s[i],y[i])*np.dot(s[i].T,q)-beta)*s[i]
        end_time = time.time()
        duration += end_time - start_time
        d=-r
        x_0=x_1
        iter+=1
    duration_per_iteration=duration/iter
    print("total:{} each:{}".format(duration,duration_per_iteration))
    return x_0,norm_list,iter,acc_list,duration,duration/iter

def AGM(a,b,x,Lambda,sparse,a1,b1):

    alpha_k_minus = 1
    yita = 0.05 #variable (0,1)

    tol = 1e-4  # vary
    x_minus = x
    xk = x

    tk_minus = 2
    tk = 1

    xk_list = []
    xk_list.append(xk)
    norm_list=[]
    num_iteration = 0
    acc_list=[]
    gradient = Gradient(a,b, xk,Lambda,sparse)
    # norm_list.append(norm_s(gradient) )
    duration=0
    while norm_s(gradient) > tol and num_iteration<10000:
        start_time = time.time()
        beta_k = (tk_minus - 1) / tk
        y = xk + beta_k * (xk - x_minus)
        alpha_k = alpha_k_minus
        x_minus = xk
        xk_ba = y - alpha_k * Gradient(a,b,y,Lambda,sparse)
        while obj_function(a,b,xk_ba,Lambda,sparse)-obj_function(a,b,y,Lambda,sparse) > -0.5*alpha_k*np.square(norm_s(Gradient(a,b,y,Lambda,sparse))):    #estimate Lipschitz
            alpha_k = yita*alpha_k
            alpha_k_minus = alpha_k
            xk_ba = y - alpha_k * Gradient(a,b,y,Lambda,sparse)
        xk = xk_ba
        xk_list.append(xk)

        tk_minus = tk
        tk = 1 / 2 * (1 + np.sqrt(1 + 4 * tk ** 2))

        gradient = Gradient(a,b,xk,Lambda,sparse)
        norm=norm_s(gradient)
        norm_list.append(norm)
        num_iteration = num_iteration + 1
        acc = accuracy(xk, a1, b1, sparse)
        acc_list.append(acc)
        end_time = time.time()
        duration += end_time - start_time
        print('iteration:{} loss:{} norm:{} accuracy:{}'.format(num_iteration, obj_function(a, b,xk, Lambda, sparse), norm,acc))
    return xk, norm_list, num_iteration, acc_list, duration,duration/num_iteration

def mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[0]  # m是样本数

    mini_batches = []  # 用来存放一个一个的mini_batch
    permutation = list(np.random.permutation(m))  # 打乱标签
    shuffle_X = X[permutation, :]  # 将打乱后的数据重新排列
    shuffle_Y = Y[permutation, :]

    num_complete_minibatches = int(m // mini_batch_size)  # 样本总数除以每个batch的样本数量
    for i in range(num_complete_minibatches):
        mini_batch_X = shuffle_X[i * mini_batch_size:(i + 1) * mini_batch_size, :]
        mini_batch_Y = shuffle_Y[i * mini_batch_size:(i + 1) * mini_batch_size, :]
        m = mini_batch_X.shape[0]
        mini_batch_X= hstack((mini_batch_X, np.ones((m, 1))))
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)


    if m % mini_batch_size != 0:
        # 如果样本数不能被整除，取余下的部分
        mini_batch_X = shuffle_X[num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = shuffle_Y[num_complete_minibatches * mini_batch_size, :]
        m = mini_batch_X.shape[0]
        mini_batch_X = hstack((mini_batch_X, np.ones((m, 1))))
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def sgd_method(a,b,x,batch_size,Lambda,sparse,a1,b1):
    xk = x
    # batchsize = 256
    epoch=0
    norm_list=[]
    acc_list=[]
    iteration=0
    duration=0
    while iteration<20000:
        data_batch=mini_batches(a,b,batch_size)
        for i in range(len(data_batch)):
            start_time = time.time()
            batch_a=data_batch[i][0]
            # print(np.shape(batch_a))
            batch_b=data_batch[i][1]
            # print(np.shape(batch_b))
            # print(batch_b.size)
            grad=Gradient(batch_a,batch_b,xk,Lambda,sparse)
            norm=norm_s(grad)
            norm_list.append(norm)
            acc = accuracy(xk, a1, b1, sparse)
            acc_list.append(acc)
            print('epoch:{} iteration:{} norm:{} accuracy:{}'.format(epoch,iteration, norm,acc))
            if norm <= 10 ** (-4) or acc>=0.8:
                return xk, norm_list, iteration, acc_list, duration,duration/(iteration)
            # eta = 0.01
            # beta = 0.01 / np.log(iteration+5)
            xk = xk - (10/(1+0.01*iteration) )* grad
            # xk = xk - 0.01* grad
            iteration+=1
            end_time = time.time()
            duration+=end_time-start_time
        epoch+=1
    return xk, norm_list, iteration, acc_list, duration,duration/(iteration)

def accuracy(x_0,a,b,sparse):
    n,m=np.shape(a)
    if sparse:
        linear=a*x_0
    else:
        linear=np.dot(a.T,x_0)
    sigmoid=1/(1+np.exp(-linear))
    # print(sigmoid)
    q=[]
    for i in range(len(sigmoid)):
        if sigmoid[i]>1/2:
            q_i=1
        else:
            q_i=-1
        q.append(q_i)
    q=np.asarray(q).reshape(-1,1)
    # print(q)
    return np.sum(np.abs(q+b))/(2*n)

def plot_division(c1,c2,x_0,path):
    plt.xlim(xmax = 50, xmin = 0)
    plt.ylim(ymax = 50, ymin = 0)
    plt.scatter(c1[0], c1[1], s=np.pi / 3, c='#DC143C')
    plt.scatter(c2[0], c2[1], s=np.pi / 3, c='#00CED1')
    x1 = np.arange(0, 50, 1)
    x2 = (-x_0[2] - x_0[0] * x1) / x_0[1]
    plt.plot(x1, x2)
    plt.savefig(path)
    plt.show()

def gradient_plot(num_iteration,grad_norm,method):
    iteration = list(i for i in range(1, num_iteration + 1))
    grad_norm=np.log(np.asarray(grad_norm))
    # norm_grad = [np.log(i) for i in norm_grad]
    # plt.figure(1,figsize=(8,10))
    plt.plot(iteration,grad_norm,linewidth=2,label=method)
    plt.xlabel('number of iteration')
    plt.ylabel('$log||gradient||$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(method+"_convergence",dpi=300)
    plt.show()

# print(a)
if __name__ == '__main__':
    a = load.c1
    c1 = load.c1_1
    c2 = load.c1_2
    b = load.label1
    n, m = np.shape(a)
    a = np.vstack((a, np.ones((1, m))))
    b = b.reshape(-1, 1)
    Lambda = 0.1
    x=np.zeros((3,1))
    a1=a
    b1=b
    m=5
    x_0,grad_norm,iteration,acc_list,_,_=sgd_method(a,b,x,1000,Lambda,False,a1,b1)
    # Draw.plot_division(c1,c2,x_0,'synthetic_4.png','L_BFGS')




