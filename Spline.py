import matplotlib.pyplot as plt 
import numpy as np
from sympy import diff, symbols, cos, sin

print("Start point") 
q0 = int(input())
print("Finish point")
qf = int(input())
print("Time")
tf = int(input())
# time = np.arange(0, tf, 0.1)
points = []
speed = []
t = int
# q0 = 2
# qf = 15
# tf = 15

t0 = 0
dotq0 = V0 = diff(q0) / tf
dotqf = Vf = diff(qf) / tf
ddotq0 = alpha0 = diff(V0) / tf
ddotqf = alphaf = diff(Vf) / tf
time = np.arange(0, tf, 0.1)

# qf = q1
# V = (q1 - q0)/(tf) * 1.5
# tb = (q0 - q1 + V * tf)/V
# alpha = V/tb 

# a0 = q0
# a1 = V0 =  diff(t0)
# V1 = diff(tf)
# a2 = (3*(q1 - q0) - (2*V0 + V1) * (tf - t0)) / (tf - t0) ** 2
# a3 = (2 * (q0 - q1) + (V0 + V1) * (tf - t0)) / (tf - t0) ** 3
# def n5 (q0, qf):
#     dotq0 = V0 = diff(q0) / tf
#     dotqf = Vf = diff(qf) / tf
#     ddotq0 = alpha0 = diff(V0) / tf
#     ddotqf = alphaf = diff(Vf) / tf
#     Y = np.array([q0, qf, dotq0, dotqf, ddotq0, ddotqf])
#     A = np.array([
#     [1, t0, t0**2, t0**3, t0 ** 4, t0**5],
#     [1, tf, tf**2, tf**3, tf ** 4, tf ** 5],
#     [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
#     [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
#     [0, 0, 2, 6 * t0, 12*t0**2, 20*t0**3],
#     [0, 0, 2, 6 * tf, 12*tf**2, 20*tf**3]
#     ])
#     X = np.dot(np.linalg.inv(A), Y)
#     a0 = X[0]
#     a1 = X[1]
#     a2 = X[2] 
#     a3 = X[3]
#     a4 = X[4]
#     a5 = X[5]
#     for t in time:
#         q = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
#         V = a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
#     points.append([q])
#     speed.append([V])
#     LSPB = np.array(points)
#     Velocity = np.array(speed)
#     return points, speed
Y = np.array([q0, qf, dotq0, dotqf, ddotq0, ddotqf])
A = np.array([
    [1, t0, t0**2, t0**3, t0 ** 4, t0**5],
    [1, tf, tf**2, tf**3, tf ** 4, tf ** 5],
    [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
    [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
    [0, 0, 2, 6 * t0, 12*t0**2, 20*t0**3],
    [0, 0, 2, 6 * tf, 12*tf**2, 20*tf**3]
    ])
X = np.dot(np.linalg.inv(A), Y)
a0 = X[0]
a1 = X[1]
a2 = X[2] 
a3 = X[3]
a4 = X[4]
a5 = X[5]
print("A = ", A)
print("X = ", X)
print("a0 = ", a0)
print("inv A = ", np.linalg.inv(A))
print ("V0 = ", V0)
print ("Vf = ", Vf)

for t in time:
    q = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    V = a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
    points.append([q])
    speed.append([V])


# LSPB = np.array(points)
# Velocity = np.array(speed)

# n5 (q0, qf)

LSPB = np.array(points)
Velocity = np.array(speed)
x = time
y = LSPB

plt.plot(x, y)
plt.plot(x, Velocity)
plt.show()