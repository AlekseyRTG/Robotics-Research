import matplotlib.pyplot as plt 
import numpy as np

print("Start point") 
q0 = int(input())
print("Finish point")
q1 = int(input())
print("Time")
tf = int(input())
# q0 = 0
# q1 = 10
#time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# tf = 50
time = np.arange(0, tf, 0.1)

qf = q1
t0 = 0

V = (q1 - q0)/(tf) * 1.5
print(V)
tb = (q0 - q1 + V * tf)/V
alpha = V/tb 

points = []

#tb = (q0 - q1 + V * tf)/V
#a = V/tb 
#V = (q1 - q0)/(tf) * 1.5
for  t in time:
    if t <= tb:
        q = q0 + alpha/2 * t * t
    if t <= tf - tb and t > tb:
        q = ((qf + q0 - V * tf) / 2) + V * t
    if t > tf - tb:
        q = qf - ((alpha * tf * tf) / 2) + alpha * tf * t - (alpha / 2) * t * t
    points.append([q])
    
    
# # for t in range (tb, tf - tb):
# #    q = ((qf + q0 - V * tf) / 2) + V * t
#     points.append([q])
    
# for t in range (tf - tb, tf):
#     q = qf - ((alpha * tf * tf) / 2) + alpha * tf * t - (alpha / 2) * t * t
#     points.append([q])
LSPB = np.array(points)
print(LSPB)

x = time
y = LSPB

plt.plot(x, y)
plt.show()