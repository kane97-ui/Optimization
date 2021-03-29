import numpy as np
import matplotlib.pyplot as plt
import time

def f1(x):
    x = x.reshape(x.size)
    return 3 + x[0] + ((1 - x[1]) * x[1] - 2) * x[1]
def f2(x):
    x = x.reshape(x.size)
    return 3 + x[0] + (x[1] - 3) * x[1]
def f(x):
    x = x.reshape(x.size)
    return f1(x) * f1(x) + f2(x) *f2(x)
def df(x):
    x = x.reshape(x.size)
    grad = np.zeros(2).reshape(2,1)
    grad[0] = 2 * f1(x) + 2 * f2(x)
    grad[1] = 2 * f1(x) * (2*x[1] - 3*(x[1]**2) - 2) + 2 * f2(x) * (2*x[1] - 3)
    return grad
def Hessian(x):
    x = x.reshape(x.size)
    hessian = np.zeros((2,2))
    hessian[0][0] = 4
    hessian[0][1] = 8*x[1]-6*(x[1]**2)-10
    hessian[1][0] = 8*x[1]-6*(x[1]**2)-10
    hessian[1][1] = 2*f1(x)*(-6*x[1]+2) + 2*(2*x[1]-3*(x[1]**2)-2)**2+4*f2(x)+2*(2*x[1]-3)**2
    return hessian
def norm(x):
    x = x.reshape(x.size)
    return np.sqrt(x[0]**2 + x[1]**2)

color_list = ['red','blue', 'green', 'orange','cyan', 'purple', 'black','yellow', 'teal',
              'coral','brown', 'darkred']
Number_iterations = []

def plot_contour():
    X = np.arange(-17.5,12.5,0.05)
    Y = np.arange(-3, 3, 0.05)
    X,Y = np.meshgrid(X,Y)
    Z = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = []
            x.append(X[i][j])
            x.append(Y[i][j])
            x = np.array(x)
            Z[i][j] = f(x)
    plt.contourf(X, Y, Z, 30, alpha=0.3, cmap=plt.cm.hot)
    plt.contour(X, Y, Z, 30, colors='grey')
    
def plot_line(xk_list, subfig_num):
    plt.figure(1)
    x = []
    y = []
    for i in range(xk_list.shape[0]):
        x.append(xk_list[i][0][0])
        y.append(xk_list[i][1][0])
    plt.plot(x,y, color = color_list[subfig_num-1], linewidth=1.5)
    plt.scatter(x, y, s=3, color='black')

def plot_convergence(y, subfig_num):
    plt.figure(2, figsize=(8,10))
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, color = color_list[subfig_num-1], linewidth=2)
    plt.xlabel('number of iteration')
    plt.ylabel('$log||(x^k-x^*)||$')
    plt.tight_layout()

def check_dir(dk_tocheck, xk, gradient, hessian, beta1, beta2):
    dk_norm = norm(dk_tocheck)
    factor1 = np.dot(gradient.T, dk_tocheck)[0][0] < 0
    facotr2 = -(np.dot(gradient.T, dk_tocheck)[0][0]) >= beta1 * np.min([1, dk_norm**beta2]) * dk_norm**2
    if factor1 == True and facotr2 == True:
        return True
    else:
        return False
def Global_Newton(initial, subfig_num):
    #paramaters
    s = 1
    sigma = 0.5
    gamma = 0.1
    beta1 = 1e-6
    beta2 = 0.1
    tol = 1e-8
    xk_list = []
    xk_xstar_list = []


    xk = initial
    num_iteration = 0
    xk_list.append(xk)

    
    gradient = df(xk)
    while norm(gradient) > tol:
        # deteriminate the direction
        hessian = Hessian(xk)
        dk_tocheck = np.linalg.solve(hessian, -gradient)
        good_dir = check_dir(dk_tocheck, xk, gradient, hessian, beta1, beta2)
        if(good_dir == False):
            dk = -gradient
        else:
            dk = dk_tocheck 
        
        alphak = s
        while True:
            if f(xk + alphak*dk) - f(xk) <= gamma * alphak * (np.dot(gradient.T, dk)[0][0]):
                break
            alphak = alphak * sigma
        xk = xk + alphak * dk
        xk_list.append(xk)
        
        gradient = df(xk)
        num_iteration = num_iteration + 1
    
    Number_iterations.append(num_iteration)
    
    plt.figure(1)
    plt.scatter(xk_list[-1][0], xk_list[-1][1], s=60, marker='*', 
                facecolors ='none', edgecolor= color_list[subfig_num-1])
    xk_list = np.array(xk_list)
    plot_line(xk_list, subfig_num)

    xstar_x1 = xk_list[-1][0][0]
    xstar_x2 = xk_list[-1][1][0]
    xstar = np.array([round(xstar_x1), xstar_x2]).reshape(2, 1)
    for i in range(xk_list.shape[0]):
        xk_xstar_list.append(norm(xk_list[i] - xstar))
    xk_xstar_list = np.array(xk_xstar_list)
    plot_convergence(xk_xstar_list, subfig_num)
# main begin

x1 = np.arange(-10, 11, 4)
x2 = np.arange(-2, 3, 4)

plt.figure(1, figsize=(10, 5))
plot_contour()
subfig_num = 1

time_list = []
for i in range(6):
    for j in range(2):
        initial = np.zeros(2).reshape(2,1)
        initial[0][0] = x1[i]
        initial[1][0] = x2[j]
        plt.figure(1)
        plt.scatter(initial[0], initial[1], s=40, marker='s', 
                    facecolors ='none', edgecolor= color_list[subfig_num-1])
        
        start = time.time()
        Global_Newton(initial, subfig_num)
        end = time.time()
        
        time_list.append(end - start)
        subfig_num = subfig_num + 1

plt.figure(1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-17, 12)
plt.ylim(-2.5, 2.5)
plt.savefig('A4_1_a', dpi=700)  

plt.figure(2)
plt.savefig('A4_1_a_convergence', dpi=700)  

print()
print('Number of iterations from different initial points:', Number_iterations)
print('Average number of iterations from different initial points:', 
      sum(Number_iterations)/len(Number_iterations))

print('Calculating time from different initial points:', time_list)
print('Average calculating time from different initial points:', 
      sum(time_list)/len(time_list))



"""
x1, x2 = symbols('x1 x2', real=True)
ans1 = diff(2*(3 + x1 + ((1 - x2) * x2 - 2) * x2) + 2*(3 + x1 + (x2 - 3) * x2), x1).subs({x1:-7, x2:1})
ans2 = diff(2*(3 + x1 + ((1 - x2) * x2 - 2) * x2) + 2*(3 + x1 + (x2 - 3) * x2), x2).subs({x1:-7, x2:1})
ans3 = diff(2*(3 + x1 + ((1 - x2) * x2 - 2) * x2)*(2*x2-3*x2**2-2) + 2*(3 + x1 + (x2 - 3) * x2)*(2*x2-3), x1).subs({x1:-7, x2:1})
ans4 = diff(2*(3 + x1 + ((1 - x2) * x2 - 2) * x2)*(2*x2-3*x2**2-2) + 2*(3 + x1 + (x2 - 3) * x2)*(2*x2-3), x2).subs({x1:-7, x2:1})
print(ans1)
print(ans2)
print(ans3)
print(ans4)
"""
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    







