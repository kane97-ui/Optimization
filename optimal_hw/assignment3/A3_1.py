from math import *
def f(x):
    return x**2/10-2*sin(x)
def g(x):
    return 1/e**x-cos(x)
def g_(x):
    return -1/e**x+sin(x)
def golden_section(function,initial_l,initial_r,tol,theta=0.382):
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
        if function(xl_)<function(xr_):
            xr=xr_
        else:
            xl = xl_
        if (xr-xl)<tol:
            print("iteration{}:xl={}, xr={}, x={},f(x) is {}".format(i,xl,xr,(xr+xl)/2,function((xr+xl)/2)))
            return (xr+xl)/2
        print("iteration{}:xl={}, xr={}, f(x) is {}".format(i,xl,xr,function((xr+xl)/2)))
        i+=1
golden_section(f,0,4,0.00005)
def Bisection(initial_l,initial_r,tol):
    xl=initial_l
    xr=initial_r
    i=1
    while True:
        xm = (xr + xl) / 2
        if g_(xm)==0:
            return xm
        if g_(xm)>0:
            xr=xm
        else:
            xl=xm
        if abs(xr-xl)<tol:
            print("iteration{}:xl={}, xr={}, x={},f(x) is {}".format(i, xl, xr,(xr + xl) / 2, g((xr + xl) / 2)))
            return (xr+xl)/2
        print("iteration{}:xl={}, xr={}, f(x) is {}".format(i,xl,xr,g((xr+xl)/2)))
        i += 1


golden_section(g,0,1,0.00005)
Bisection(0,1,0.00005)
