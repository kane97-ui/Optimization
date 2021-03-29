import numpy as np
import Data_preparation as data
import time
import logistic_regression as LR

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

def BackTracking(a,b,x,Lambda,delta,sparse,a1,b1):
    s=1
    sigma = 0.5
    gamma = 0.1
    tol =  1e-3
    xk_list = []
    xk = x
    grad=Gradient(a,xk,Lambda,delta,b,sparse)
    num_iteration = 0 
    xk_list.append(xk)
    norm_list=[]
    norm=norm_s(grad)
    norm_list.append(norm)
    acc_list=[]
    duration=0
    while norm_s(grad) > tol and num_iteration<10000:
        start_time=time.time()
        alphak = s
        dk = -Gradient(a,xk,Lambda,delta,b,sparse)
        while True:
            if obj_function(a,xk + alphak*dk,delta,Lambda,b,sparse) - obj_function(a,xk,delta,Lambda,b,sparse) <= gamma * alphak * np.dot(Gradient(a,xk,Lambda,delta,b,sparse).T, dk):
                break
            alphak = alphak * sigma
        xk = xk + alphak * dk 
        xk_list.append(xk)
        gradient = Gradient(a,xk,Lambda,delta,b,sparse)
        # print(gradient)
        ng= norm_s(gradient)
        norm_list.append(ng)
        num_iteration = num_iteration + 1
        acc=accuracy(xk,a1,b1,sparse)
        acc_list.append(acc)
        print('iteration:{} loss:{} norm:{} accuracy:{}'.format(num_iteration, obj_function(a,xk, delta,Lambda, b,sparse), ng, acc))
        end_time=time.time()
        duration += end_time - start_time
    return xk,norm_list,num_iteration,acc_list,duration,duration/num_iteration

def AGM(a,b,x,Lambda,delta,sparse,a1,b1):

    alpha_k_minus = 1
    yita = 0.05 #variable (0,1)

    tol = 1e-3  # vary
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
    while norm_s(gradient) > tol and num_iteration<10000:
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


def G_BFGS(a,b,x,Lambda,delta,sparse,a1,b1):
    def S(x_1, x_0):
        return x_1 - x_0

    def Y(x_1, x_0):
        return Gradient(a, x_1, Lambda, delta,b,sparse) - Gradient(a, x_0, Lambda, delta,b,sparse)
    n, num = np.shape(a)
    x_k = x
    m, n = np.shape(x_k)
    H_k = np.eye(m)
    d_k = -np.dot(H_k, Gradient(a,x_k,Lambda,delta,b,sparse))
    iter = 1
    acc_list=[]
    norm_list=[]
    gradient = Gradient(a,x_k,Lambda,delta,b,sparse)
    norm_list.append(norm_s(gradient) )
    duration=0
    while iter<100:
        start_time = time.process_time()
        alpha = backtracking(a,b,Lambda,0.5, 0.1, delta,d_k, x_k,sparse)
        x_k1 = x_k + alpha * d_k
        norm = norm_s(Gradient(a,x_k1,Lambda,delta,b,sparse))
        acc = accuracy(x_k1, a1, b1, sparse)
        acc_list.append(acc)
        norm_list.append(norm)
        print('iteration:{} loss:{} norm:{} accuracy:{}'.format(iter, obj_function(a, x_k1, delta,Lambda,b, sparse),norm, acc))
        if norm <= 10 ** (-3):
            break
        s_k = S(x_k1, x_k)
        y_k = Y(x_k1, x_k)
        if np.dot(s_k.T, y_k) <= 0:
            H_k1 = H_k
        else:
            A1 = np.dot(s_k - np.dot(H_k, y_k), s_k.T)
            a2 = np.dot(s_k, (s_k - np.dot(H_k, y_k)).T)
            a3 = np.dot((s_k - np.dot(H_k, y_k)).T, y_k)
            B1 = np.dot(s_k.T, y_k)
            H_k1 = H_k + (A1 + a2) / B1 + a3 * np.dot(s_k, s_k.T) / B1 ** 2
        d_k = - np.dot(H_k1, Gradient(a,x_k1,Lambda,delta,b,sparse))
        iter += 1
        end_time = time.process_time()
        duration += end_time - start_time
        x_k = x_k1
    return x_k1,norm_list,iter,acc_list,duration,duration/iter

def store(li,value,limitation):
    if len(li)<limitation:
        li.append(value)
    else:
        li.append(value)
        li=li[1:]
    return li

def L_BFGS(a,b,x,m,Lambda,delta,sparse,a1,b1):
    def S(x_1, x_0):
        return x_1 - x_0

    def Y(x_1, x_0):
        return Gradient(a, x_1, Lambda, delta, b, sparse) - Gradient(a, x_0, Lambda, delta, b, sparse)

    def P(s, y):
        return 1 / np.dot(s.T, y)

    def H_0(s, y):
        m, n = np.shape(s)
        return np.dot(s.T, y) / np.square(norm_s(y) )* np.eye(m)
    if sparse:
        num, n = np.shape(a)
    else:
        n, num = np.shape(a)
    x_0=x
    d = -Gradient(a,x_0,Lambda,delta,b,sparse)
    s=[]
    y=[]
    norm_list=[]
    iter=1
    duration=0
    acc_list=[]
    while iter<10000:
        start_time=time.time()
        alpha=backtracking(a,b,Lambda,0.5, 0.1, delta,d, x_0,sparse)
        # print(alpha)
        x_1 = x_0 + alpha * d
        acc=accuracy(x_1,a1,b1,sparse)
        acc_list.append(acc)
        norm=norm_s(Gradient(a,x_1,Lambda,delta,b,sparse))
        norm_list.append(norm)
        print('iteration:{} loss:{} norm:{} accuracy:{}'.format(iter, obj_function(a, x_1, delta,Lambda,b, sparse), norm,acc))
        if norm<= 10 ** (-3):
            break
        s0 = S(x_1, x_0)
        y0 = Y(x_1,x_0)
        if np.dot(s0.T,y0)<=10**(-14):
            h=np.eye(n)
        else:
            h=H_0(s0,y0)
        q=Gradient(a,x_1,Lambda,delta,b,sparse)
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




def backtracking(a, b, Lambda, sigma, gama, delta,direction, x, sparse):
    alpha = 1
    while obj_function(a,x + alpha * direction,delta,Lambda,b,sparse) - obj_function(a, x,delta,Lambda,b,sparse) > gama * alpha * (np.dot(Gradient(a,x, Lambda,delta,b,sparse).T, direction)):
        alpha = alpha * sigma
    return alpha
def accuracy(x_0,a,b,sparse):
    n,m=np.shape(b)
    if sparse:
        linear=a*x_0
    else:
        linear=np.dot(a.T,x_0)
    # print(sigmoid)
    q=[]
    for i in range(len(linear)):
        if linear[i]>0:
            q_i=1
        else:
            q_i=-1
        q.append(q_i)
    q=np.asarray(q).reshape(-1,1)
    # print(q)
    np.sum(np.abs(q + b)) / (2 * n)

    return np.sum(np.abs(q+b))/(2*n)
if __name__ == '__main__':
    a = data.c4
    n,m=np.shape(a)
    a= np.vstack((a, np.ones((1, m))))
    # print(np.shape(c))
    b= data.label4
    b=b.T
    delta = 0.0001
    lamda = 0.1
    m=5
    x = np.random.random(3).reshape(3,1)
    L_BFGS(a,b,x,m,lamda,delta,False,a,b)

