import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy
from sympy import lambdify, hessian, ordered
from mpl_toolkits import mplot3d

###########################
# define function f(x, y) #
###########################
def np_func(x, y):
    return np.sin(x + y - 1) + (x - y - 1)**2 - 1.5 * x + 2.5 * y + 1

def sp_func(x, y):
    return sympy.sin(x + y - 1) + (x - y - 1) ** 2 - 1.5 * x + 2.5 * y + 1

############
# plotting #
############
fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
x = np.arange(-1.0, 5.0, 0.05)
y = np.arange(-3.0, 4.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array(np_func(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#############################
# get random number of x, y #
#############################
np.random.seed()
rand_x = np.random.uniform(low=-1.0, high=5.0, size=1)
rand_y = np.random.uniform(low=-3.0, high=4.0, size=1)

#######################################
# get gradient and hessian of f(x, y) #
#######################################
x, y = sympy.symbols('x y')
f = sp_func(x, y)
f_diff_by_x = sympy.diff(f, x)
f_diff_by_y = sympy.diff(f, y)
fx_diff_by_x = sympy.diff(f_diff_by_x, x)
fx_diff_by_y = sympy.diff(f_diff_by_x, y)
fy_diff_by_x = sympy.diff(f_diff_by_y, x)
fy_diff_by_y = sympy.diff(f_diff_by_y, y)
print('f_diff_by_x:', f_diff_by_x)
print('f_diff_by_y:', f_diff_by_y)
print('fx_diff_by_x:', fx_diff_by_x)
print('fx_diff_by_y:', fx_diff_by_y)
print('fy_diff_by_x:', fy_diff_by_x)
print('fy_diff_by_y:', fy_diff_by_y)

s = (x, y)
v = list(ordered(f.free_symbols))
H = hessian(f, v)
g_func_x = lambdify(s, f_diff_by_x, modules='numpy')
g_func_y = lambdify(s, f_diff_by_y, modules='numpy')
h_func = lambdify(s, H, modules='numpy')

############################
# plot gradient of f(x, y) #
############################
xs, ys = np.meshgrid(np.linspace(-1.0, 5.0), np.linspace(-3.0, 4.0))
zs = [float(f.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs.ravel(), ys.ravel())]
zs = np.array(zs).reshape(xs.shape)

plt.figure(2, figsize=(12, 8))
cb = plt.contour(xs, ys, zs, 5, levels=np.logspace(-2.0, 1.5, 20), cmap=cm.rainbow)
plt.colorbar(cb)

xs_q, ys_q = np.meshgrid(np.linspace(-1.0, 5.0), np.linspace(-3.0, 4.0))
xs_q_grad = [-float(f_diff_by_x.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs_q.ravel(), ys_q.ravel())]
ys_q_grad = [-float(f_diff_by_y.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs_q.ravel(), ys_q.ravel())]

plt.quiver(xs_q, ys_q, xs_q_grad, ys_q_grad, width=0.001, scale=100, color='red')

############################
# init figure3 for 2D plot #
############################
plt.figure(3, figsize=(12, 8))
plt.contour(xs, ys, zs, 5, levels=np.logspace(-2.0, 1.5, 20), cmap=cm.rainbow)

###################################################################################
# do gradient descent optimization (begin, end points are black color else green) #
###################################################################################
l = 0.01
xk_x = rand_x
xk_y = rand_y
count = 0
ax.scatter(xk_x, xk_y, np_func(xk_x, xk_y), color='black')
plt.scatter(xk_x, xk_y, color='black')
print('global minima = ', zs.min())
print('start : (', xk_x, ', ', xk_y, ')')
while True:
    xk1_x = xk_x - l * g_func_x(xk_x, xk_y)
    xk1_y = xk_y - l * g_func_y(xk_x, xk_y)
    if np.sqrt((xk1_x - xk_x)**2 + (xk1_y - xk_y)**2) < 0.0001:
        print('count : ', count)
        print('minima = f(', xk1_x, ',', xk1_y, ') = ', np_func(xk1_x, xk1_y))
        ax.scatter(xk1_x, xk1_y, np_func(xk1_x, xk1_y), color='black')
        plt.scatter(xk1_x, xk_y, color='black')
        break
    ax.scatter(xk1_x, xk1_y, np_func(xk1_x, xk1_y), color='green')
    plt.scatter(xk1_x, xk_y, color='green')
    xk_x = xk1_x
    xk_y = xk1_y
    count += 1

##################################################################
# do newton optimization (begin, end points are black else blue) #
##################################################################
xk_x = float(rand_x)
xk_y = float(rand_y)
count = 0
print('start : (', xk_x, ', ', xk_y, ')')
while True:
    h_inv = np.linalg.inv(h_func(xk_x, xk_y))
    w, v = np.linalg.eig(h_inv)
    absw = np.array([[abs(w[0]), 0], [0, abs(w[1])]])
    abs_h_inv = v.dot(absw.dot(v.transpose()))

    xk1_x = xk_x - (abs_h_inv[0][0] * g_func_x(xk_x, xk_y) + abs_h_inv[0][1] * g_func_y(xk_x, xk_y))
    xk1_y = xk_y - (abs_h_inv[1][0] * g_func_x(xk_x, xk_y) + abs_h_inv[1][1] * g_func_y(xk_x, xk_y))
    if np.sqrt((xk1_x - xk_x)**2 + (xk1_y - xk_y)**2) < 0.00001:
        print('count : ', count)
        print('minima = f(', xk1_x, ',', xk1_y, ') = ', np_func(xk1_x, xk1_y))
        ax.scatter(xk1_x, xk1_y, np_func(xk1_x, xk1_y), color='black')
        plt.scatter(xk1_x, xk_y, color='black')
        break
    ax.scatter(xk1_x, xk1_y, np_func(xk1_x, xk1_y), color='blue')
    plt.scatter(xk1_x, xk_y, color='blue')
    xk_x = xk1_x
    xk_y = xk1_y
    count += 1

plt.show()
