import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plot_pred_vs_act = 0
plot_loss_vs_epoch = 0
def polynomial(x, coefficients,degree):
    result = sum(coefficients[i] * x**i for i in range(degree + 1))
    return result

def g(x): 
    result = 0.6*x+0.1
    return result

points = [[0.1, 0.19], [0.2, 0.18], [0.3, 0.36], [0.4, 0.33]]

# Separate x and y values
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

degrees = [1, 2, 4, 8, 16, 32, 64]
x = torch.tensor(x_values,requires_grad = False) # fitting parameters
y = torch.tensor(y_values,requires_grad = False) # fitting parameters

if(plot_pred_vs_act==1):
    plt.figure(1)
    y_plot = y.detach().numpy()
    x_plot = x.detach().numpy()
    plt.scatter(x_plot, y_plot, label='actual')
    
test_loss_l = []
test_loss_std_dev = []
test_loss_mean = []
training_loss_mean = []
n=0
for degree in degrees : # For example, fitting a quadratic polynomial
    
    for _ in range(10):
        training_loss = []
        a = torch.rand([degree+1],requires_grad = True) # fitting parameters
        # print(a,degree)

        # optimizer
        opt = torch.optim.Adam([a],lr=0.01)
        k_epoch=[]
        loss_epoch=[]
        loss = 0
        for k in range(20000):
            opt.zero_grad() # an important step, don't forget
            # print(x)
            yp = polynomial(x,a,degree) # do the prediction
            # print(yp)
        
            loss = torch.mean((y-yp)*(y-yp)) # mean square error
            # print(loss)
            loss.backward() # use of torch.autograd
            opt.step() # perform one gradient descend step
            if k%1000==0:
                print(k,'loss ',loss.item())
                if(plot_loss_vs_epoch==1):
                    k_epoch.append(k)
                    loss_epoch.append(loss.item())
        training_loss.append(loss.item())
        test_loss = 0
        parameter = a.detach().numpy()
        for del_x in range(-500,500,1) : # For example, fitting a quadratic polynomial
            fx = polynomial(del_x/100,parameter,degree)
            gx = g(del_x/100)
            test_loss = test_loss + (pow((fx - gx),2)*0.01)
        test_loss_l.append(test_loss)
    training_loss_mean.append(np.mean(training_loss))
    test_loss_std_dev.append(np.std(test_loss_l))
    test_loss_mean.append(np.mean(test_loss_l))

plt.figure(2)
plt.title("Training Loss (Mean) vs Degree")
plt.xlabel("Degree")
plt.ylabel("Training Loss (Mean)")
plt.grid(True)
plt.plot(degrees, training_loss_mean)
plt.legend()

plt.figure(3)    
# Plot the mean test loss as a line plot
plt.plot(degrees, test_loss_mean, marker='o', color='black', label='Test Loss mean')
# Calculate upper and lower bounds for the standard deviation
upper_bounds = np.array(test_loss_mean) + np.array(test_loss_std_dev)
lower_bounds = np.array(test_loss_mean) - np.array(test_loss_std_dev)
# Fill the area between the upper and lower bounds with a shaded color
plt.fill_between(degrees, lower_bounds, upper_bounds, color='grey', alpha=0.5, label='Test Loss Std Dev')
plt.xlabel("Degree")
plt.ylabel("Test Loss(Mean)")
plt.title("Test Loss(Mean) vs Degree")
plt.grid(True)

# Add a legend to distinguish between the mean and std dev bounds
plt.legend()

plt.tight_layout()
plt.show()














# plt.figure(4)
# plt.scatter(degrees, test_loss)
# plt.legend()
# plt.show()

# plt.legend()
# plt.show()
exit()        
# Generate some synthetic data with added noise
torch.manual_seed(0)
x_train = torch.linspace(-2, 2, 100).unsqueeze(1)  # Create input data
y_true = polynomial_function(x_train) + torch.randn(x_train.shape) * 2  # Add noise to generate target data

# Define a polynomial regression model using nn.Linear
class PolynomialRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Instantiate the polynomial regression model
input_dim = 1  # One input feature (x)
output_dim = 1  # One output (y)
model = PolynomialRegression(input_dim, output_dim)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_true)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Test the model by predicting values
x_test = torch.linspace(-2, 2, 100).unsqueeze(1)  # Test data
predicted_values = model(x_test)

# Plot the original data, true polynomial, and predicted polynomial
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_true, label='Original Data', s=10, color='blue')
plt.plot(x_test, polynomial_function(x_test), label='True Polynomial', linewidth=2, color='green')
plt.plot(x_test, predicted_values.detach().numpy(), label='Predicted Polynomial', linewidth=2, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression')
plt.show()

# Print the final loss
print("Final Loss:", losses[-1])

exit()
def g(x): # linear example
    result = 0.6*x+0.1
    return result
degree = 2
x=2

powers = [x ** d for d in range(degree+1)]
print(powers)
exit()
# Data
points = [[0.1, 0.19], [0.2, 0.18], [0.3, 0.36], [0.4, 0.33]]

# Separate x and y values
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

# # Plot the points
# plt.scatter(x_values, y_values, color='blue', marker='o')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Scatter Plot of Points')
# plt.grid(True)
# plt.show()
training_loss = []
test_loss=[]
degree_range = [1, 2, 4, 8, 16, 32, 64]
for degree in degree_range : # For example, fitting a quadratic polynomial
    coefficients = np.polyfit(x_values, y_values, degree)
    polynomial = np.poly1d(coefficients)
    fitted_y = polynomial(x_values)
    training_loss.append( sum(pow((fitted_y - y_values),2)))
    test_loss_D = 0
    for del_x in range(-500,500,1) : # For example, fitting a quadratic polynomial
        fx = polynomial(del_x/100)
        gx = g(del_x/100)
        test_loss_D = test_loss_D + (pow((fx - gx),2)*(del_x/100))
    test_loss.append(test_loss_D)    
print (training_loss,test_loss) 
        
plt.plot(degree_range, training_loss, label="training_loss")
plt.plot(degree_range, test_loss, label="test_loss")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Polynomial Fit')
plt.show()

# exit()

# for loop in range(0,4,1):
#     sum = (g(x_values[loop])-y_values[loop])**2
    
# for loop in range(-5,5,1):
#     sum2 = (g(loop)-y_values[loop])**2
    
# print (sum,sum2
    



