import numpy as np
import matplotlib.pyplot as plt
import time


def norm(x):
    x = x.reshape(x.size)
    return np.sqrt(np.sum(x**2))

def norm_square(x):
    x = x.reshape(x.size)
    return np.sum(x**2)

def huber(t):
    if np.abs(t) <= delta:
        return (t**2) / (2*delta)
    else:
        return np.abs(t) - delta/2

def d_huber(t):
    if np.abs(t) <= delta:
        return t/delta
    else:
        return t/np.abs(t)

def log_fun(t):
    return np.log(1 + t**2/v) 

def d_log_fun(t):
    return (1/(1+t**2/v)) * (2*t/v)
    
def phi_1(x):
    return norm_square(x)

def d_phi_1(x):
    grad = 2*x
    return grad

def phi_2(x):
    x = x.reshape(x.size)
    Sum = 0
    for xi in x:
        Sum = Sum + huber(xi)
    return Sum

def d_phi_2(x):
    x = x.reshape(x.size)
    grad = np.zeros((x.size,1))
    for i in range(grad.size):
        grad[i] = d_huber(x[i])
    return grad

    
def phi_3(x):
    x = x.reshape(x.size)
    Sum = 0
    for xi in x:
        Sum = Sum + log_fun(xi)
    return Sum

def d_phi_3(x):
    x = x.reshape(x.size)
    grad = np.zeros((x.size,1))
    for i in range(grad.size):
        grad[i] = d_log_fun(x[i])
    return grad
def f1(x):
    return 1/2 * norm_square(np.dot(A,x) - b) + mu * phi_1(x)

def f2(x):
    return 1/2 * norm_square(np.dot(A,x) - b) + mu * phi_2(x)

def f3(x):
    return 1/2 * norm_square(np.dot(A,x) - b) + mu * phi_3(x)

def df1(x):
    return np.dot(A.T, np.dot(A, x)-b) + mu * d_phi_1(x)
    
def df2(x):
    return np.dot(A.T, np.dot(A, x)-b) + mu * d_phi_2(x)

def df3(x):
    return np.dot(A.T, np.dot(A, x)-b) + mu * d_phi_3(x)

color_list = ['brown', 'green', 'red', 'blue', 'cyan', 'purple', 'yellow', 'orange', 'teal',
              'coral', 'darkred', 'black']
def plot_convergence(y, method, subfig_num, tol, knownL):
    plt.figure(2-knownL)
    
    if knownL == True:
        sum_fig = 2
    else:
        sum_fig = 3
    plt.subplot(sum_fig, 1, subfig_num)
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, label = method, color = color_list[subfig_num], linewidth=2)
    plt.legend()
    plt.xlabel('number of iteration (tol={})'.format(tol))
    plt.ylabel('$log||gradient||$')
    plt.tight_layout()

def plt_compare_and_sparse(x_solution, subfig_num, knownL):
    if knownL == True:
        Type = 'knwonL'
        fignum_begin = 3
    else:
        Type = 'UnknownL'
        fignum_begin = 5
        
    plt.figure(fignum_begin + subfig_num - 1, figsize=(8,20))
    plt.subplot(2,1,1)
    plt.scatter(x_solution, x_star, color=color_list[3], s=2)
    xmin = x_solution.min()
    xmax = x_solution.max()
    
    plt.plot([x_star.min(), x_star.max()], [x_star.min(), x_star.max()], '--', color='red', linewidth=1, label='diagonal line')
    plt.xlim(xmin-0.02, xmax+0.01)
    plt.xlabel('Solution')
    plt.ylabel('$x^*$')
    plt.legend()
    
    plt.subplot(2,1,2)
    n = x_solution.size
    plt.plot([0,n], [0,0], '--', color='red', linewidth=1)
    plt.scatter(np.arange(n)+1, x_solution, color=color_list[1], s=2)
    plt.xlabel('$i$ (the $i^{th}$ unit of solution)')
    plt.ylabel('The value of $i^{th}$ unit in solution')

    plt.tight_layout()
    plt.savefig('compare_f' +str(subfig_num)+'_'+Type, dpi=700)
    
def IGM_Known_L(inital, smooth_func_type):
    if smooth_func_type == 1:
        df = df1
        L = L1
        method = 'IGM method on $f_1$ with known L'
    elif smooth_func_type == 2:
        df = df2
        L = L2
        method = 'IGM method on $f_2$ with known L'
    
    tol = 1e-4
    beta = 0.5
    alpha = 1.99 * (1-beta) / L
    
    x_minus = initial
    xk = initial
    xk_list = []
    xk_list.append(xk)
    norm_gradient_list = []
    num_iteration = 0
    
    gradient = df(xk)
    norm_gradient_list.append(norm(gradient))
    while norm(gradient) > tol:
        y = xk + beta * (xk - x_minus)
        x_minus = xk
        xk = y - alpha * df(xk)
        
        xk_list.append(xk)
        
        gradient = df(xk)
        norm_gradient_list.append(norm(gradient))

        num_iteration  = num_iteration + 1
        
        
    xk_list = np.array(xk_list)
    print(xk_list.shape)
    x_solution = xk_list[-1]
    
    print('norm of (xk-x_star):', norm(x_solution - x_star))
    norm_gradient_list = np.array(norm_gradient_list)
    knownL = True
    plot_convergence(norm_gradient_list, method, smooth_func_type, tol, knownL)
    plt_compare_and_sparse(x_solution, smooth_func_type, knownL)
    
def IGM_Unknown_L(initial, smooth_func_type):
    if smooth_func_type == 1:
        df = df1
        f = f1
        method = 'IGM method on $f_1$ with unknown L'
    elif smooth_func_type == 2:
        df = df2
        f = f2
        method = 'IGM method on $f_2$ with unknown L'
    else:
        df = df3 
        f = f3
        method = 'IGM method on $f_3$ with unknown L'
        
    tol = 1e-4  # vary
    beta = 0.5  # vary
    l = 1  #vary
    alpha = 1.99 * (1-beta) / l
    
    x_minus = initial
    xk = initial
    
    xk_list = []
    xk_list.append(xk)
    
    num_iteration = 0
    norm_gradient_list = []
    gradient = df(xk)
    norm_gradient_list.append(norm(gradient))

    while norm(gradient) > tol:
        y = xk + beta * (xk - x_minus)
        xk_bar = y - alpha * df(xk)
        while f(xk_bar) - f(xk) > np.dot(df(xk).T, xk_bar-xk) + l/2 * norm_square(xk_bar-xk):
            l = 2 * l 
            alpha = 1.99 * (1-beta) / l
            xk_bar = y - alpha * df(xk)
        
        x_minus = xk
        xk = xk_bar
        xk_list.append(xk)
        
        gradient = df(xk)
        norm_gradient_list.append(norm(gradient))

        num_iteration  = num_iteration + 1
    
    xk_list = np.array(xk_list)
    print(xk_list.shape)
    x_solution = xk_list[-1]
    
    print('norm of (xk-x_star):', norm(x_solution - x_star))
    norm_gradient_list = np.array(norm_gradient_list)
    knownL = False
    plot_convergence(norm_gradient_list, method, smooth_func_type, tol, knownL)
    plt_compare_and_sparse(x_solution, smooth_func_type, knownL)
    

# main begin    
#parameters

np.random.seed(2222)
n = 3000
m = 300
s = 30
delta = 1e-3
v = 1e-4

mask = np.random.choice(np.arange(1,n+1), s, replace=False)
x_star = np.zeros((n,1))
for i in range(n):
    if i+1 in mask:
        x_star[i][0] = np.random.randn(1)[0]
A = np.random.randn(m, n)
c = 0.01 * np.random.randn(m, 1)
b = np.dot(A, x_star) + c



initial = np.zeros((n,1))

print('----------Known L----------')

# for known L: f1 and f2
# plt.figure(1, figsize=(8, 12))
# mu = 1
# L1 = 2*mu + np.linalg.norm(np.dot(A.T, A), ord = 2)
# start = time.time()
# IGM_Known_L(initial, 1)
# end = time.time()
# print('f1 time:', end-start)
#
# mu = 1
# L2 = mu*(1/delta) + np.linalg.norm(np.dot(A.T, A), ord = 2)
# start = time.time()
# IGM_Known_L(initial, 2)
# end = time.time()
# print('f2 time:', end-start)
#
# plt.figure(1)
# plt.savefig('KnownL.png', dpi=700)


print('----------Unknown L----------')

# for unknown L: f1, f2, f3
plt.figure(2, figsize=(8, 18))

mu = 1
start = time.time()
IGM_Unknown_L(initial, 1)
end = time.time()
print('f1 time:', end-start)

mu = 1
start = time.time()
IGM_Unknown_L(initial, 2)
end = time.time()
print('f2 time:', end-start)

mu = 0.1
start = time.time()
IGM_Unknown_L(initial, 3)
end = time.time()
print('f3 time:', end-start)

plt.figure(2)
plt.savefig('UnknownL.png', dpi=700)

    
    
    
    
    
    
    
    
    
    
    
    