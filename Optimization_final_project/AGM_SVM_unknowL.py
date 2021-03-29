import numpy as np
import Data_preparation as data
import time

def norm_s(x):
    return np.sqrt(np.sum(np.square(x)))
def huber(t,delta):
    m,n=np.shape(t)
    n_list=np.zeros((m,1))
    n_list[t>delta]=t[t>delta]-delta/2
    n_list[(t>0)&(t<delta)]=np.square(t[(t>0)*(t<delta)])/ (2*delta)
    n_list[t<=0]=0
    return n_list

def obj_function(a,x,delta,Lambda,b,sparse):
    if sparse:
        linear = a * x
    else:
        linear = np.dot(a.T, x)
    active = 1 + np.multiply(-b, linear)
    return np.sum(huber(active,delta)) + np.square(norm_s(x) )* Lambda / 2

def Gradient(a,x,Lambda,delta,b,sparse):
    if sparse:
        linear = a * x
        n, m = np.shape(a)
    else:
        linear = np.dot(a.T, x)
        m, n = np.shape(a)
    active = 1 + np.multiply(-b, linear)
    one=np.ones(n)
    one=one.reshape(-1,1)
    index0=(active>delta)
    one[index0]=1
    index1 = (active<= 0)
    one[index1]= 0
    index2 = (active > 0) & (active<delta)
    one[index2] = active[index2]/delta
    if sparse:
        grad_x = Lambda * x + (-a).T*(one * b)
    else:
        grad_x = Lambda * x + np.dot(-a, one*b)
    return grad_x

def AGM(a,b,x,Lambda,delta,sparse,a1,b1):

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
    gradient = Gradient(a,xk,Lambda,delta,b,sparse)
    norm_list.append(norm_s(gradient) )
    duration=0
    while norm_s(gradient) > tol:
        start_time = time.time()
        beta_k = (tk_minus - 1) / tk
        y = xk + beta_k * (xk - x_minus)
        alpha_k = alpha_k_minus
        x_minus = xk
        xk_ba = y - alpha_k * Gradient(a,y,Lambda,delta,b,sparse)
        while obj_function(a,xk_ba,delta,Lambda,b,sparse)-obj_function(a,y,delta,Lambda,b,sparse) > -0.5*alpha_k*np.square(norm_s(Gradient(a,y,Lambda,delta,b,sparse))):    #estimate Lipschitz
            alpha_k = yita*alpha_k
            alpha_k_minus = alpha_k
            xk_ba = y - alpha_k * Gradient(a,y,Lambda,delta,b,sparse)
        xk = xk_ba
        xk_list.append(xk)

        tk_minus = tk
        tk = 1 / 2 * (1 + np.sqrt(1 + 4 * tk ** 2))

        gradient = Gradient(a,xk,Lambda,delta,b,sparse)
        norm=norm_s(gradient)
        norm_list.append(norm)
        num_iteration = num_iteration + 1
        acc = accuracy(xk, a1, b1, sparse)
        acc_list.append(acc)
        end_time = time.time()
        duration += end_time - start_time
        print('iteration:{} loss:{} norm:{} accuracy:{}'.format(num_iteration, obj_function(a, xk, Lambda, delta,b, sparse), norm,acc))
    return xk, norm_list, num_iteration, acc_list, duration,duration/num_iteration


# main begin
# parameters
def accuracy(x_0,a,b,sparse):
    n,m=np.shape(b)
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

if __name__ == '__main__':
    a = data.c1
    n, m = np.shape(a)
    a = np.vstack((a, np.ones((1, m))))
    # print(np.shape(c))
    b = data.label1
    b = b.T
    delta = 0.0001
    lamda = 0.1
    x= np.random.random((3, 1))
    AGM(a,b,x,lamda,delta,False,a,b)








