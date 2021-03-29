import matplotlib.pyplot as plt
import numpy as np
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

color_list = ['blue', 'green', 'red', 'cyan', 'purple', 'yellow', 'orange', 'teal',
              'coral', 'darkred', 'brown', 'black']
def plot_convergence(y, method, subfig_num, tol):
    plt.figure(1)
    plt.subplot(2,1,subfig_num)
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, label = method, color = color_list[subfig_num], linewidth=2)
    plt.legend()
    plt.xlabel('number of iteration (tol={})'.format(tol))
    plt.ylabel('$log||gradient||$')
    plt.tight_layout()

def plt_compare_and_sparse(x_solution, subfig_num):
    plt.figure(subfig_num+1, figsize=(8,20))
    plt.subplot(2,1,1)
    plt.scatter(x_solution, x_star, color=color_list[0], s=2)
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
    plt.savefig('compare_f' +str(subfig_num), dpi=700)
    
    
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

def AGM(initial, smooth_func_type):
    if smooth_func_type == 1:
        df = df1
        alpha_k = 1 / L1
        method = 'AGM method on $f_1$'
    elif smooth_func_type == 2:
        df = df2
        alpha_k = 1 / L2
        method = 'AGM method on $f_2$'
        
    tol = 1e-4  # vary
    x_minus = initial
    xk = initial
    
    tk_minus = 1
    tk = 1
    
    xk_list = []
    xk_list.append(xk)
    norm_gradient_list = []
    num_iteration = 0
     
    gradient = df(xk)
    norm_gradient_list.append(norm(gradient))
    
    while norm(gradient) > tol:
        beta_k = (tk_minus - 1)/tk
        y = xk + beta_k * (xk - x_minus)
        
        x_minus = xk
        xk = y - alpha_k * df(y)
        xk_list.append(xk)
        
        tk_minus = tk
        tk = 1/2 * (1 + np.sqrt(1+4*tk**2))
        
        gradient = df(xk)
        norm_gradient_list.append(norm(gradient))
        # print(norm(gradient))
        num_iteration  = num_iteration + 1
    
    xk_list = np.array(xk_list)
    print(xk_list.shape)
    norm_gradient_list = np.array(norm_gradient_list)
    plot_convergence(norm_gradient_list, method, smooth_func_type, tol)
    
    x_solution = xk_list[-1]
    print('norm of (xk-x_star):', norm(x_solution - x_star))
    plt_compare_and_sparse(x_solution, smooth_func_type)
    

# main begin    
#parameters

np.random.seed(2222)
n = 3000
m = 300
s = 30
mu = 1
delta = 1e-3
v = 1e-5

A = np.random.randn(m, n)
mask = np.random.choice(np.arange(1,n+1), s, replace=False)
x_star = np.zeros((n,1))
for i in range(n):
    if i+1 in mask:
        x_star[i][0] = np.random.randn(1)[0]

b = np.dot(A, x_star) + 0.01 * np.random.randn(m, 1)
L1 = 2*mu + np.linalg.norm(np.dot(A.T, A), ord = 2)
L2 = mu*(1/delta) + np.linalg.norm(np.dot(A.T, A), ord = 2)

initial = np.zeros((n,1))

plt.figure(1, figsize=(8,12))

print('----------f1----------')
start = time.time()
AGM(initial, 1)
end = time.time()
print('time:', end - start)
print()

print('----------f2----------')
start = time.time()
AGM(initial, 2)
end = time.time()
print('time:', end - start)
print()

plt.figure(1)
plt.savefig('4_2_a_convergence.png', dpi=700)





