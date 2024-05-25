

import torch.optim
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

def g(x): # linear example
    term1 = 1 - 1 / (1 + torch.exp(- (10 * x - 2)))
    term2 = 1 - 1 / (1 + torch.exp(10 * x + 2))
    result = term1 * term2
    return result

def f(a,x): # linear example
    result=0
    n=0
    for j in range(-3,4,1):
        for k in range (-3,4,1):
            result += a[n]*g(pow(2,j)*(x-k))
            n+=1
            # print (n,j,k)
    # print (result)   
    return result

if __name__ == '__main__':
 poly_deg = 1
 a = torch.ones([49],requires_grad = True) # fitting parameters
 x = torch.tensor(np.loadtxt('2-2-x.txt'),requires_grad = False) # fitting parameters
 y = torch.tensor(np.loadtxt('2-2-f_x.txt'),requires_grad = False) # fitting parameters
 
 # optimizer
 opt = torch.optim.SGD([a],lr=0.1)
 for k in range(20):
    opt.zero_grad() # an important step, don't forget
    # print(x)
    yp = f(a,x) # do the prediction
    # print(yp)
 
    loss = torch.mean((y-yp)*(y-yp)) # mean square error
    # print(loss)
    loss.backward() # use of torch.autograd
    opt.step() # perform one gradient descend step
    if k%10000==0:
        print(k,'loss ',loss)
parameter = np.round(a.detach().numpy(),decimals=4)
print('a',parameter)

y_plot = y.detach().numpy()
yp_plot = yp.detach().numpy()
x_plot = x.detach().numpy()
parameter=[ 0.8777,-0.5362,-0.7903,-0.4449,-0.0962,0.8342,2.587,0.6223,-0.6315
,-0.5128,0.4516,1.455,-0.9731,0.788,3.8921,-0.8462,0.2855,-0.1155
,2.1305,-2.1442,0.3202,1.0636,-0.6593,-2.7781,-1.188,0.3664,-0.2408
,-0.662,-0.1065,-0.5175,-3.9714,0.4302,0.7829,0.7598,-2.6225,-0.0985
,-1.2938,-1.0989,1.3297,0.9153,-0.4074,-2.1012,0.5914,-0.4646,1.035
,0.4746,0.1881,-0.8493,0.3693]
parameter_round = np.round(parameter,decimals=4)
yp_round_2 = f(parameter_round,x) # do the prediction
yp_round=yp_round_2.detach().numpy()
# Plot Line 1
plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_plot, label='actual', color='blue', linestyle='-', marker='o')

# Plot Line 2
plt.plot(x_plot, yp_plot, label='predition', color='green', linestyle='--', marker='s')
# Plot Line 2
# plt.plot(x_plot, yp_round, label='predition', color='black', linestyle='-.', marker='o')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Two Line Graphs')
plt.legend()

# Show the graph
plt.show()

# import torch.optim
# import torch
# import numpy as np
# import math

# def g(x): # linear example
#     term1 = 1 - 1 / (1 + torch.exp(- (10 * x - 2)))
#     term2 = 1 - 1 / (1 + torch.exp(10 * x + 2))
#     result = term1 * term2
#     return result

# def f(a,x): # linear example
#     result=0
#     for j in range(-3,4,1):
#         for k in range (-3,4,1):
#                 result += a*g(pow(2,j)*(x-k))
#     # print ('haha')
#     # print (result)   
#     return result

# if __name__ == '__main__':
#  poly_deg = 1
#  a = torch.ones([1],requires_grad = True) # fitting parameters
#  x = torch.tensor(np.loadtxt('2-1-x.txt'),requires_grad = False) # fitting parameters
#  y = torch.tensor(np.loadtxt('2-1-f_x.txt'),requires_grad = False) # fitting parameters
#  print(a)
#  print(x)
#  print(y)
 
#  # optimizer
#  opt = torch.optim.SGD([a],lr=0.1)
#  for k in range(32):
#     opt.zero_grad() # an important step, don't forget
#     # print(x)
#     yp = f(a,x) # do the prediction
#     # print(yp)
 
#     loss = torch.mean((y-yp)*(y-yp)) # mean square error
#     # print(loss)
#     loss.backward() # use of torch.autograd
#     opt.step() # perform one gradient descend step
#     if k%1==0:
#         print(k,'loss ',loss)
#         print(k,'a',a)

# a [ 0.9937  0.0866 -0.2287 -0.2866 -0.3821 -0.2722  0.1495  0.0628  0.0497
#   0.6365  0.2714  0.3679  0.0962  0.1868 -0.02    0.0242  1.5653  0.2007
#   0.2553  0.3428  0.0212  0.0253 -0.1073  0.3853 -0.124   0.0764 -0.1198
#  -0.0157  0.0038  0.064  -0.1992  0.0404  0.785   0.0167  0.3275 -0.0464
#  -0.0444 -0.0101 -0.0283  0.2106 -0.014  -0.7576  0.0459  0.0302  0.0775
#   0.0305 -0.0825  0.0321 -2.3813]


# [ 0.9937 , 0.0866 ,-0.2287, -0.2866 ,-0.3821 ,-0.2722 , 0.1495 , 0.0628 , 0.0497,
#   0.6365 , 0.2714 , 0.3679 , 0.0962 , 0.1868 ,-0.02,0.0242,1.5653,0.2007,
# 0.2553,0.3428,0.0212,0.0253, -0.1073,0.3853, -0.124, 0.0764, -0.1198,
#  -0.0157,0.0038,0.064,-0.1992,0.0404,0.785, 0.0167,0.3275, -0.0464,
#  -0.0444, -0.0101, -0.0283,0.2106, -0.014,-0.7576,0.0459,0.0302,0.0775,
# 0.0305, -0.0825,0.0321 ,-2.3813]
# a [ 0.8777,-0.5362,-0.7903,-0.4449,-0.0962,0.8342,2.587,0.6223,-0.6315
# ,-0.5128,0.4516,1.455,-0.9731,0.788,3.8921,-0.8462,0.2855,-0.1155
# ,2.1305,-2.1442,,0.3202,1.0636,-0.6593,-2.7781,-1.188,0.3664,-0.2408
# ,-0.662,,-0.1065,-0.5175,-3.9714,,0.4302,0.7829,0.7598,-2.6225,-0.0985
# ,-1.2938,-1.0989,1.3297,0.9153,-0.4074,-2.1012,,0.5914,-0.4646,1.035
# ,0.4746,0.1881,-0.8493,0.3693]
# [ 0.8777,-0.5362,-0.7903,-0.4449,-0.0962,0.8342,2.587,0.6223,-0.6315
# ,-0.5128,0.4516,1.455,-0.9731,0.788,3.8921,-0.8462,0.2855,-0.1155
# ,2.1305,-2.1442,,0.3202,1.0636,-0.6593,-2.7781,-1.188,0.3664,-0.2408
# ,-0.662,,-0.1065,-0.5175,-3.9714,0.4302,0.7829,0.7598,-2.6225,-0.0985
# ,-1.2938,-1.0989,1.3297,0.9153,-0.4074,-2.1012,,0.5914,-0.4646,1.035
# ,0.4746,0.1881,-0.8493,0.3693]