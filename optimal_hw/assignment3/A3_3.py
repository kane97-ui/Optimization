from math import *
# from A3_1 import *
def f1(x1,x2):
    return  3 + x1 + ((1 - x2) * x2 - 2) * x2
def f2(x1,x2):
    return  3 + x1 + (x2 - 3) * x2
def f(f1,f2,x1,x2):
    return f1(x1,x2)**2+f2(x1,x2)**2
def gradient(f1,f2,x1,x2):
    grad=[]
    grad.append(2*f1(x1,x2)+2*f2(x1,x2))
    grad.append(2*f1(x1,x2)*((2*x2)-3*(x2**2)-2)+2*f2(x1,x2)*(2*x2-3))
    return grad
def func_alpha(f1,f2,x1,x2,alpha,dire):
    return f1(x1+alpha*dire[0], x2+alpha*dire[1]) ** 2 + f2(x1+alpha*dire[0], x2++alpha*dire[1]) ** 2
def golden_section(function,x1,x2,dire,initial_l,initial_r,tol,theta=0.382):
    """

    :param function: the optimal function we are going to call
    :param initial_l: the initial xl
    :param initial_r: the initial xr
    :param theta: default 0.382
    :param tol: a value which can estimate whether it gets a good iteration
    :return:
    """
    xl=initial_l
    xr=initial_r
    i=1
    while True:
        xl_=theta*xr+(1-theta)*xl
        xr_ = theta * xl + (1 - theta) * xr
        if function(f1,f2,x1,x2,xl_,dire)<function(f1,f2,x1,x2,xr_,dire):
            xr=xr_
        else:
            xl = xl_
        if (xr-xl)<tol:
            # print("iteration{}:xl={}, xr={}, x={},f(x) is {}".format(i,xl,xr,(xr+xl)/2,function(f1,f2,x1,x2,(xr+xl)/2,dire)))
            return (xr+xl)/2
        # print("iteration{}:xl={}, xr={}, f(x) is {}".format(i,xl,xr,function(f1,f2,x1,x2,(xr+xl)/2,dire)))
        i+=1
def direction(grad):
    dire=[]
    dire.append(-grad[0])
    dire.append(-grad[1])
    return dire
def backtracking(sigma,gama,tol,initial_point):
    ir=1
    save_xk=[]
    save_norm_grad=[]
    x1=initial_point[0]
    x2=initial_point[1]
    save_xk.append((x1,x2))
    grad=gradient(f1,f2,x1,x2)
    print(grad)
    while True:
        alpha = 1
        dire = direction(grad)
        while f(f1,f2,x1+alpha*dire[0],x2+alpha*dire[1])-f(f1,f2,x1,x2)>gama*alpha*(grad[0]*dire[0]+grad[1]*dire[1]):
            alpha=alpha*sigma
        x1=x1+alpha*dire[0]
        x2=x2+alpha*dire[1]
        function_value=f(f1,f2,x1,x2)
        grad = gradient(f1, f2, x1, x2)
        norm_grad = (grad[0] ** 2 + grad[1] ** 2)**0.5
        print("iteration{}:xk={},norm_grad={},f(x) = {}".format(ir,[x1,x2],norm_grad,function_value))
        save_xk.append((x1,x2))
        save_norm_grad.append(norm_grad)
        if norm_grad<tol:
            break
        ir += 1
    return save_xk,save_norm_grad,ir

# backtracking(0.5,0.1,1e-5,[0,0])
# grad=gradient(f1,f2,0,0)
# dire=direction(grad)
# print(golden_section(func_alpha,0,0,dire,0,2,1e-6,theta=0.382))

def exact_line_search(tol,initial_point):
    ir = 1
    save_xk=[]
    save_norm_grad=[]
    x1 = initial_point[0]
    x2 = initial_point[1]
    save_xk.append((x1,x2))
    grad=gradient(f1,f2,x1,x2)
    # print(grad)
    while True:
        dire = direction(grad)
        alpha=golden_section(func_alpha,x1,x2,dire,0,2,1e-6,theta=0.382)
        x1 = x1 + alpha * dire[0]
        x2 = x2 + alpha * dire[1]
        function_value = f(f1, f2, x1, x2)
        grad = gradient(f1, f2, x1, x2)
        norm_grad = (grad[0] ** 2 + grad[1] ** 2) ** 0.5
        print("iteration{}:xk={},norm_grad={},f(x) = {}".format(ir, [x1, x2], norm_grad, function_value))
        save_xk.append((x1,x2))
        save_norm_grad.append(norm_grad)
        if norm_grad < tol:
            break
        ir += 1
    return save_xk, save_norm_grad, ir
# exact_line_search(1e-5,[0,0])

def dimishing_step(tol,initial_point):
    ir = 1
    save_xk = []
    save_norm_grad = []
    x1 = initial_point[0]
    x2 = initial_point[1]
    save_xk.append((x1, x2))
    grad = gradient(f1, f2, x1, x2)
    while True:
        dire = direction(grad)
        alpha=0.01/log(ir+12)
        x1 = x1 + alpha * dire[0]
        x2 = x2 + alpha * dire[1]
        function_value = f(f1, f2, x1, x2)
        grad = gradient(f1, f2, x1, x2)
        norm_grad = (grad[0] ** 2 + grad[1] ** 2) ** 0.5
        print("iteration{}:xk={},norm_grad={},f(x) = {}".format(ir, [x1, x2], norm_grad, function_value))
        save_xk.append((x1, x2))
        save_norm_grad.append(norm_grad)
        if norm_grad < tol:
            break
        ir += 1
    return save_xk, save_norm_grad, ir
backtracking(0.5,0.1,1e-5,[0,0])
exact_line_search(1e-5,[0,0])
dimishing_step(1e-5,[0,0])















