        
import math
import matplotlib.pyplot as plt


n = 0.3
x = 0

# Given parameters
B1 = 0.9
B2 = 0.999
alpha = 0.3
epsilon = 0

# Initialize variables
x = 0.0  # Initial value of 'x'
m = 0.0
v = 0.0
t = 0

# Initialize 'h' to zero
h = 0.0
parameter = 0 
t_t = []
g_t = []
m_t = []
v_t = []
mhat_t = []
vhat_t = []
x_t = []
# Optimization loop
while parameter<=0:
    t += 1
    if x < 1:
         gradient = -1
    elif x >= 1 : 
        gradient = 1

    # Update first and second moments
    m = B1 * m + (1 - B1) * gradient
    v = B2 * v + (1 - B2) * gradient ** 2

    # Bias correction
    mhat = m / (1 - B1 ** t)
    vhat = v / (1 - B2 ** t)

    # Update 'x' using Adam optimizer
    parameter = alpha * mhat / (math.sqrt(vhat) + epsilon)
    g_t.append(gradient)
    m_t.append(m)
    v_t.append(v)
    mhat_t.append(mhat)
    vhat_t.append(vhat)
    x_t.append(x)
    t_t.append(t)
    if parameter>0:
        break
    
    x = x - parameter
    print(parameter,x)
    

# Round 'h' to 2 decimal places
h = x - 1
h = round(h, 2)

print("Maximum height 'h' of the bump:", h)

# Create the first figure with a 2x2 grid of subplots
plt.figure(figsize=(10, 8))  # You can adjust the figure size as needed

plt.subplot(2, 3, 1)
# Plot your data with specific parameters for this subplot
plt.plot(t_t,g_t)
plt.title('gt vs t')

plt.subplot(2, 3, 2)
# Plot your data with specific parameters for this subplot
plt.plot(t_t,m_t)
plt.title('mt vs t')

# Create the third subplot
plt.subplot(2, 3, 3)
# Plot your data with specific parameters for this subplot
plt.plot(t_t,v_t)
plt.title('vt vs t')

# Create the third subplot
plt.subplot(2, 3, 4)
# Plot your data with specific parameters for this subplot
plt.plot(t_t,mhat_t)
plt.title('mhat vs t')

# Create the third subplot
plt.subplot(2, 3, 5)
# Plot your data with specific parameters for this subplot
plt.plot(t_t,vhat_t)
plt.title('vhat vs t')

# Create the third subplot
plt.subplot(2, 3, 6)
# Plot your data with specific parameters for this subplot
plt.plot(t_t,x_t)
plt.title('x vs t')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show or save the figure
plt.show()

