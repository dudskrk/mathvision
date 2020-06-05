import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import ordered, Matrix, hessian
from sympy import lambdify
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D

def func(x, y):
    return (x + y) * (x * y + x * y ** 2)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
x = np.arange(-1.0, 1.5, 0.05)
y = np.arange(-1.2, 0.2, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array(func(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

x, y = sympy.symbols('x y')
f = func(x, y)
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

v = list(ordered(f.free_symbols))
H = hessian(f, v)
s = (x, y)
h_func = lambdify(s, H, modules='numpy')
h_0 = h_func(0, 0)
h_1 = h_func(0, -1)
h_2 = h_func(1, -1)
h_3 = h_func(3/8, -3/4)
print(h_0)
print(h_1)
print(h_2)
print(h_3)

v0 = np.linalg.eigvals(h_0)
v1 = np.linalg.eigvals(h_1)
v2 = np.linalg.eigvals(h_2)
v3 = np.linalg.eigvals(h_3)
print('f(0, 0)', v0)
print('f(0, -1)', v1)
print('f(1, -1)', v2)
print('f(3/8, -3/4)', v3)

xs, ys = np.meshgrid(np.linspace(-1.0, 1.5), np.linspace(-1.2, 0.2))
zs = [float(f.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs.ravel(), ys.ravel())]
zs = np.array(zs).reshape(xs.shape)

plt.figure(2, figsize=(12, 8))
plt.contour(xs, ys, zs, 5, levels=np.logspace(-1.2, 2.3, 20), cmap=plt.cm.rainbow)

xs_q, ys_q = np.meshgrid(np.linspace(-1.0, 1.5), np.linspace(-1.2, 0.2))
xs_q_grad = [float(f_diff_by_x.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs_q.ravel(), ys_q.ravel())]
ys_q_grad = [float(f_diff_by_y.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs_q.ravel(), ys_q.ravel())]

plt.quiver(xs_q, ys_q, xs_q_grad, ys_q_grad, width=0.001, scale=50, color='red')
plt.show()
