# 题干中的huber-type version
def huber(t,delta):
    m,n=np.shape(t)
    n_list=np.zeros((m,1))
    n_list[t>delta]=t[t>delta]-delta/2
    n_list[(t>0)&(t<delta)]=np.square(t[(t>0)*(t<delta)])/ (2*delta)
    n_list[t<=0]=0
    # print(n_list)
    return n_list

# def huber(t):
#     n_list=[]
#     for i in t:
#         if i<= delta:
#             n_list.append(np.max([0,i]))**2 / (2*delta)
#         else:
#             n_list.append(i - delta/2)
#     return np.asarray(n_list)

def norm_s(x):
    return np.sqrt(np.sum(np.square(x)))

def obj_function(a,x,Lambda,b):
    linear = np.dot(a.T, x)
    active = 1 + np.multiply(-b, linear)
    return np.sum(huber(active,delta)) + np.square(norm_s(x) )* Lambda / 2

# def obj_function(x):
#     linear=np.dot(a.T,x)
#     active=1 + np.multiply(-b,linear)
#     return np.sum(huber(active)) + norm_func(x) * Lambda /2

def gradient(a,x,Lambda,delta,b):
    m, n = np.shape(a)
    linear = np.dot(a.T, x)
    # print(linear)
    active = 1 + np.multiply(-b, linear)
    # print(active)
    # print(np.shape(active))
    # active0=np.ma.array(active,active>delta)
    # print(active)
    one=np.ones(n)
    one=one.reshape(-1,1)
    index0=(active>delta)
    one[index0]=1

    index1 = (active<= 0)
    one[index1]= 0

    index2 = (active > 0) & (active<delta)
    one[index2] = active[index2]/delta
    # print(one)
    grad_x = Lambda * x + np.dot(-a, one*b)

    return grad_x
# def gradient(a,x,Lambda,delta,b):
#     m,n=np.shape(a)
#     linear = np.dot(a.T, x)
#     # print(linear)
#     active = 1 - np.multiply(-b,linear)
#     if active > delta:
#         grad_x = Lambda * x + np.dot(-a,b)
#     else:
#         if active <= 0:
#             grad_x = Lambda * x 
#         else:
#             grad_x = Lambda * x + actice/delta * np.dot(-a,b)
#     return grad_x

def S(x_1,x_0):
    return x_1-x_0

def Y(x_1,x_0):
    return gradient(a,x_1,Lambda,delta,b)-gradient(a,x_0,Lambda,delta,b)

def P(s,y):
    return 1/np.dot(s.T,y)

def H_0(s,y):
    m,n=np.shape(s)
    return np.dot(s.T,y)/norm_func(y)*np.eye(m)

def direction(x,s,y):
    return -np.dot(H_0(s_k,y_k),gradient(x))

def backtracking(sigma,gama,direction,x):
        alpha = 1
        while obj_function(a,x+alpha*direction,Lambda,b)-obj_function(a,x,Lambda,b)>gama*alpha*(np.dot(gradient(a,x,Lambda,delta,b).T,direction)):
            alpha=alpha*sigma
        return alpha

def store(li,value,limitation):
    if len(li)<limitation:
        li.append(value)
    else:
        li.append(value)
        li=li[1:]
    return li    

# sigma = 0.5
# gama = 0.1
def G_BFGS(a, x, Lambda, delta, b):
    n, num = np.shape(a)
    x_k = x
    m,n=np.shape(x_k)
#    H_0 = rho * np.eye(x_0.size)
#    rho = 0.5
    H_k = np.eye(m)
    d_k = -np.dot(H_k,gradient(a, x_k, 0.1, 0.001, b))
    s=[]
    y=[]
    iter=1
    duration=0
    while True:
        start_time=time.process_time()
        alpha=backtracking(0.5, 0.1, d_k, x_k)
        # print(alpha)
        x_k1 = x_k + alpha * d_k
        norm=np.sqrt(norm_func(gradient(a, x_k1, Lambda, delta, b)))
        if norm <= 10 ** (-4):
            break
        s_k = S(x_k1,x_k)
        y_k = Y(x_k1,x_k)
        if np.dot(s_k.T, y_k) <= 0:
            H_k1=H_k
        else:
            a1 = np.dot(s_k-np.dot(H_k, y_k),s_k.T)
            a2 = np.dot(s_k,(s_k-np.dot(H_k, y_k)).T)
            a3 = np.dot((s_k-np.dot(H_k, y_k)).T,y_k)
            b1 = np.dot(s_k.T,y_k)
            H_k1 = H_k + (a1+a2)/b1 + a3 * np.dot(s_k,s_k.T) / b1 ** 2
        
        d_k = - np.dot(H_k1,gradient(a,x_k1,Lambda,delta,b)) 
        iter += 1
        end_time = time.process_time()
        duration += end_time - start_time
        print(print('iteration:{} loss:{} norm:{} '.format(iter, obj_function(a,x_k1,Lambda,b), norm)))
        x_k=x_k1
    duration_per_iteration=duration/iter
    print("total:{} each:{}".format(duration,duration_per_iteration))
    return x_k,norm
        

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

# print(a)
delta = 0.001
x=np.random.random((3,1))
gradient(a, x, 0.1, 0.001, b)
x_0,_=G_BFGS(a, x, 0.1, 0.001, b)
# print(accuracy(x_0))
# plot_division(c1,c2,x_0,'synthetic_4.png')