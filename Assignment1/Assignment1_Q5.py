import torch
import numpy as np

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid_dz(z):
    print ((1 - 1/(1 + np.exp(-z)))/(1 + np.exp(-z)))
    print (np.exp(-z)/(1 + np.exp(-z))**2)
    return np.exp(-z)/(1 + np.exp(-z))**2

    # import sympy as sp
    # # Define the symbolic variable and the sigmoid function
    # z = sp.symbols('z')
    # sigmoid = 1 / (1 + sp.exp(-z))
    # # Compute the derivative of the sigmoid function with respect to z
    # sigmoid_derivative = sigmoid * (1 - sigmoid)
    # # Print the result
    # print(sigmoid_derivative)
    # (1 - 1/(1 + exp(-z)))/(1 + exp(-z))
    
# Given values
x = 1.0 #torch.tensor(1.0, requires_grad=False)
w1 = 0.1 #torch.tensor(0.1, requires_grad=True)
b1 = 0.0 #torch.tensor(0.0, requires_grad=True)
w2 = -0.2 #torch.tensor(-0.2, requires_grad=True)
b2 = 0.1 #torch.tensor(0.1, requires_grad=True)
w3 = -0.1 #torch.tensor(-0.1, requires_grad=True)
b3 = 0.2 #torch.tensor(0.2, requires_grad=True)
y = 1.0 #torch.tensor(1.0, requires_grad=False)

z1 =np.round( w1*x + b1 ,decimals=4)
a1 =np.round( sigmoid(z1),decimals=4)
z2 =np.round( w2*a1 + b2,decimals=4)
a2 =np.round( sigmoid(z2),decimals=4)
z3 =np.round( w3*a2 + b3 ,decimals=4)
a3 =np.round( sigmoid(z3),decimals=4)
L = np.round((a3 - y)**2,decimals=4)

print(f"a1: {a1}")
print(f"a2: {a2}")
print(f"a3: {a3}")
print(f"L: {L}")


dL_da3 = 2*(a3 - y)
da3_dz3 = dsigmoid_dz(z3)
dz3_dw3 = a2
dz3_db3 = 1
dz3_da2 = w3
da2_dz2 = dsigmoid_dz(z2)
dz2_dw2 = a1
dz2_db2 = 1
dz2_da1 = w2
da1_dz1 = dsigmoid_dz(z1)
dz1_dw1 = x
dz1_db1 = 1

dL_dw3 = np.round(dL_da3 * da3_dz3 * dz3_dw3,decimals=4)
dL_dw2 =np.round( dL_da3 * da3_dz3 * dz3_da2 * da2_dz2 * dz2_dw2 ,decimals=4)
dL_dw1 =np.round( dL_da3 * da3_dz3 * dz3_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1,decimals=4)

dL_db3 =np.round( dL_da3 * da3_dz3 * dz3_db3,decimals=4)
dL_db2 =np.round( dL_da3 * da3_dz3 * dz3_da2 * da2_dz2 * dz2_db2 ,decimals=4)
dL_db1 =np.round( dL_da3 * da3_dz3 * dz3_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_db1,decimals=4)

print(f"dL_dw1: {dL_dw1}")
print(f"dL_dw2: {dL_dw2}")
print(f"dL_dw3: {dL_dw3}")
print(f"dL_db1: {dL_db1}")
print(f"dL_db2: {dL_db2}")
print(f"dL_db3: {dL_db3}")
exit()
# Given values
x = 1.0 #torch.tensor(1.0, requires_grad=False)
w1 = 0.1 #torch.tensor(0.1, requires_grad=True)
b1 = 0.0 #torch.tensor(0.0, requires_grad=True)
w2 = -0.2 #torch.tensor(-0.2, requires_grad=True)
b2 = 0.1 #torch.tensor(0.1, requires_grad=True)
w3 = -0.1 #torch.tensor(-0.1, requires_grad=True)
b3 = 0.2 #torch.tensor(0.2, requires_grad=True)
y = 1.0 #torch.tensor(1.0, requires_grad=False)

z1 =   w1*x + b1  = x 
a1 =   σ(z1) = σ(x)
z2 =   w2*a1 + b2 = -0.2*σ(x)+0.1
a2 =   σ(z2) = σ(-0.2*σ(x)+0.1)
z3 =   w3*a2 + b3  = -0.1*σ(-0.2*σ(x)+0.1) + 0.2
a3 =   σ(z3) = σ(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)
L =   (a3 - y)**2 

dL/da3 = 2*(a3 - y)
da3/dz3 = σ'(z3) = σ'(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)
dz3/dw3 = a2
dz3/db3 = 1
dz3/da2 = w3
da2/dz2 = σ'(z2) =σ'(-0.2*σ(x)+0.1)
dz2/dw2 = a1
dz2/db2 = 1
dz2/da1 = w2
da1/dz1 = σ'(z1) = σ'(x)
dz1/dw1 = x
dz1/db1 = 1

dL/dw3 = dL/da3 * da3/dz3 * dz3/dw3 = 2*(a3 - y)*σ'(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)*a2
dL/dw2 =  dL/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/dw2 = 2*(a3 - y)*σ'(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)*w3*σ'(-0.2*σ(x)+0.1)*a1
dL/dw1 =  dL/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1 = 2*(a3 - y)*σ'(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)*w3*σ'(-0.2*σ(x)+0.1)*w2*σ'(x)*x

dL/db3 =  dL/da3 * da3/dz3 * dz3/db3 = 2*(a3 - y)*σ'(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)*1
dL/db2 =  dL/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/db2  = 2*(a3 - y)*σ'(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)*w3*σ'(-0.2*σ(x)+0.1)*1
dL/db1 =  dL/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/db1  = 2*(a3 - y)*σ'(-0.1*σ(-0.2*σ(x)+0.1) + 0.2)*w3*σ'(-0.2*σ(x)+0.1)*w2*σ'(x)*1
