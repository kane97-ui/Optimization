import numpy as np
import matplotlib.pyplot as plt
import time

def f(x): # Rosenbrock function
    x = x.reshape(x.size)
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2
def df(x):
    x = x.reshape(x.size)
    grad = np.zeros(2).reshape(2,1)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) + 2 * x[0] -2
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad
def Hessian(x):
    x = x.reshape(x.size)
    hessian = np.zeros((2,2))
    hessian[0][0] = -400 * (x[1] - 3*x[0]**2) + 2
    hessian[0][1] = -400 * x[0]
    hessian[1][0] =  -400 * x[0]
    hessian[1][1] = 200
    return hessian
def norm(x):
    x = x.reshape(x.size)
    return np.sqrt(x[0]**2 + x[1]**2)

color_list = ['red','blue', 'green', 'orange','cyan', 'purple', 'black','yellow', 'teal',
              'coral','brown', 'darkred']
label_list = ['Newton method', 'Gradient method with backtracking']
tol_list = [1e-1, 1e-3, 1e-5]


def plot_contour():
    X = np.arange(-1.51,1.6,0.05)
    Y = np.arange(-1.55, 2.55, 0.05)
    X,Y = np.meshgrid(X,Y)
    Z = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = []
            x.append(X[i][j])
            x.append(Y[i][j])
            x = np.array(x)
            Z[i][j] = f(x)
    plt.contourf(X, Y, Z, 20, alpha=0.3, cmap=plt.cm.hot)
    plt.contour(X, Y, Z, 20, colors='grey')
    
def plot_line(xk_list, subfig_num):
    plt.figure(1)
    x = []
    y = []
    for i in range(xk_list.shape[0]):
        x.append(xk_list[i][0][0])
        y.append(xk_list[i][1][0])
    plt.plot(x,y, color = color_list[subfig_num], linewidth=1.5, label=label_list[subfig_num])
    plt.scatter(x, y, s=3, color='black')

def plot_convergence(y, method, subfig_num, tol_index):
    tol_index = tol_index + 1
    plt.figure(2, figsize=(8,10))
    plt.subplot(3,1,tol_index)
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, label = method, color = color_list[subfig_num], linewidth=2)
    plt.legend()
    plt.xlabel('number of iteration (tol={})'.format(str(tol_list[tol_index-1])))
    plt.ylabel('$log||(x^k-x^*)||$')
    plt.tight_layout()
    plt.xlim(0,100)
    
def check_dir(dk_tocheck, xk, gradient, hessian, beta1, beta2):
    dk_norm = norm(dk_tocheck)
    factor1 = np.dot(gradient.T, dk_tocheck)[0][0] < 0
    facotr2 = -(np.dot(gradient.T, dk_tocheck)[0][0]) >= beta1 * np.min([1, dk_norm**beta2]) * dk_norm**2
    if factor1 == True and facotr2 == True:
        return True
    else:
        return False
def Global_Newton(initial, subfig_num, tol, tol_index):
    #paramaters
    
    xstar = np.array([1, 1]).reshape(2, 1)
    s = 1
    sigma = 0.5
    gamma = 1e-4
    beta1 = 1e-6
    beta2 = 0.1
    xk_list = []
    xk_xstar_list = []
    alphak_list_Newton = []
    Always_Use_Newton_Dir = True
    
    xk = initial
    num_iteration = 0
    xk_list.append(xk)
    xk_xstar_list.append((norm(xk-xstar)))
    
    gradient = df(xk)
    while norm(gradient) > tol:
        # deteriminate the direction
        hessian = Hessian(xk)
        dk_tocheck = np.linalg.solve(hessian, -gradient)
        good_dir = check_dir(dk_tocheck, xk, gradient, hessian, beta1, beta2)
        if(good_dir == False):
            dk = -gradient
            Always_Use_Newton_Dir = False
        else:
            dk = dk_tocheck 
        
        alphak = s
        alphak_list_Newton.append(alphak)
        
        while True:
            if f(xk + alphak*dk) - f(xk) <= gamma * alphak * (np.dot(gradient.T, dk)[0][0]):
                break
            alphak = alphak * sigma
        alphak_list_Newton.append(alphak)
        xk = xk + alphak * dk
        xk_list.append(xk)
        xk_xstar_list.append((norm(xk-xstar)))
        
        gradient = df(xk)
        num_iteration = num_iteration + 1
    
    print('tolerance:', tol)
    print('Newton_num_iteration:', num_iteration)
    print('Always_Use_Newton_Dir:', Always_Use_Newton_Dir)
    print('alpha_k of Newton method', alphak_list_Newton)
    print()
    
    method = 'Newton method'
    xk_xstar_list = np.array(xk_xstar_list)
    plot_convergence(xk_xstar_list, method, subfig_num, tol_index)
    
    if tol == 1e-5:
        plt.figure(1)
        plt.scatter(xk_list[-1][0], xk_list[-1][1], s=60, marker='*', 
                    facecolors ='none', edgecolor= 'r')
        xk_list = np.array(xk_list)
        plot_line(xk_list, subfig_num)

def gradient_method(initial, subfig_num, tol, tol_index):
    xstar = np.array([1, 1]).reshape(2, 1)
    s = 1
    sigma = 0.5
    gamma = 1e-4
    xk_list = []
    xk_xstar_list = []
    alphak_list_GM = []

    
    xk = initial
    gradient = df(xk)
    num_iteration = 0
    xk_list.append(xk)
    xk_xstar_list.append((norm(xk-xstar)))
    
    while norm(gradient) > tol:
        alphak = s
        alphak_list_GM.append(alphak)
        
        dk = -df(xk)
        while True:
            if f(xk + alphak*dk) - f(xk) <= gamma * alphak * (np.dot(df(xk).T, dk)[0][0]):
                break
            alphak = alphak * sigma
            
        alphak_list_GM.append(alphak)
        xk = xk + alphak * dk
        xk_list.append(xk)
        xk_xstar_list.append((norm(xk-xstar)))
        
        gradient = df(xk)
        num_iteration = num_iteration + 1
    
    print('tolerance:', tol)
    print('Gradient_mothd_num_iteration:', num_iteration)
    print('alpha_k of GM method', alphak_list_GM)
    print()
    
    method = 'Gradient method'
    xk_xstar_list = np.array(xk_xstar_list)
    plot_convergence(xk_xstar_list, method, subfig_num, tol_index)
    
    if tol == 1e-5:
        plt.figure(1)
        plt.scatter(xk_list[-1][0], xk_list[-1][1], s=60, marker='*', 
                facecolors ='none', edgecolor= 'r')
        xk_list = np.array(xk_list)
        plot_line(xk_list, subfig_num)
        
# main begin

x1 = np.arange(-10, 11, 4)
x2 = np.arange(-2, 3, 4)

plt.figure(1, figsize=(10, 5))
plot_contour()

initial = np.array([[-1.2], [1]])
plt.figure(1)
plt.scatter(initial[0], initial[1], s=40, marker='s', 
                    facecolors ='none', edgecolor= 'b')

print('----------Newton Method----------')
for index, tol in enumerate(tol_list):
    
    start = time.time()
    Global_Newton(initial, 0, tol, index)
    end = time.time()
    
    print('time:', end - start)
    print()
    
print('----------Gradient Method----------')
for index, tol in enumerate(tol_list):
    
    start = time.time()
    gradient_method(initial, 1, tol, index)
    end = time.time()
    
    print('time:', end - start)
    print()

plt.figure(1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-1.5, 1.5)
plt.ylim(-1, 2)
plt.legend()
# plt.show()
plt.savefig('A4_1_b', dpi=700)

plt.figure(2)
plt.savefig('convergence', dpi=700)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    







