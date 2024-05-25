import torch
import torch.nn as nn

# Load the input tensor from the file (input.pt)
input_tensor = torch.load('pixel_input.pt')  # Assuming 'input.pt' is in the current directory

# Define the MaxPool1d layer with the specified parameters
maxpool_layer = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
# Apply the MaxPool1d layer to the input tensor
output_nn = maxpool_layer(input_tensor)


def max_pool1d_manual(input_tensor, kernel_size, stride):
    batch_size, num_channels, seq_len = input_tensor.shape
    output_seq_len = (seq_len - kernel_size) // stride + 1
    
    # Initialize an empty tensor for the output
    output_tensor = torch.zeros(batch_size, num_channels, output_seq_len)
    
    for b in range(batch_size):
        for c in range(num_channels):
            for i in range(0, seq_len - kernel_size + 1, stride):
                # Extract the subsequence
                subseq = input_tensor[b, c, i:i+kernel_size]
                # Take the max value in the subsequence
                max_value = torch.max(subseq)
                # Assign the max value to the corresponding position in the output
                output_tensor[b, c, i // stride] = max_value
    
    return output_tensor

# Apply the manual max_pool1d operation to the input tensor
output_manual = max_pool1d_manual(input_tensor, kernel_size=2, stride=1)

# Check if the two tensors match (element-wise comparison)
match = torch.allclose(output_nn, output_manual, atol=1e-6)
print("Both output tensors for Maxpool match:", match)

##################################################################################
# Define the AvgPool1d layer with the specified parameters
avgpool_layer = nn.AvgPool1d(kernel_size=2, stride=1, padding=0)

# Apply the AvgPool1d layer to the input tensor
output_nn = avgpool_layer(input_tensor)

def avg_pool1d_manual(input_tensor, kernel_size, stride):
    batch_size, num_channels, seq_len = input_tensor.shape
    output_seq_len = (seq_len - kernel_size) // stride + 1
    
    # Initialize an empty tensor for the output
    output_tensor = torch.zeros(batch_size, num_channels, output_seq_len)
    
    for b in range(batch_size):
        for c in range(num_channels):
            for i in range(0, seq_len - kernel_size + 1, stride):
                # Extract the subsequence
                subseq = input_tensor[b, c, i:i+kernel_size]
                # Calculate the average of the subsequence
                avg_value = torch.mean(subseq)
                # Assign the average value to the corresponding position in the output
                output_tensor[b, c, i // stride] = avg_value
    
    return output_tensor

# Apply the manual avg_pool1d operation to the input tensor
output_manual = avg_pool1d_manual(input_tensor, kernel_size=2, stride=1)

# Check if the two tensors match (element-wise comparison)
match = torch.allclose(output_nn, output_manual, atol=1e-6)
print("Both output tensors for Avgpool match:", match)

##################################################################################

filter_tensor = torch.load('4_3_filter.pt')

import torch.nn.functional as F

# Apply the conv1d operation using torch.nn.functional
output_nn = F.conv1d(input_tensor, filter_tensor)
def conv1d_manual(input_tensor, filter_tensor, stride, padding):
    batch_size, num_channels, seq_len = input_tensor.shape
    _, _, filter_size = filter_tensor.shape
    output_seq_len = (seq_len + 2 * padding - filter_size) // stride + 1

    # Initialize an empty tensor for the output
    output_tensor = torch.zeros(batch_size, num_channels, output_seq_len)

    for b in range(batch_size):
        for c in range(num_channels):
            for i in range(0, seq_len + 2 * padding - filter_size + 1, stride):
                # Extract the subsequence from the input tensor
                subseq = input_tensor[b, c, i:i+filter_size]
                # Perform element-wise multiplication with the filter tensor
                conv_value = torch.sum(subseq * filter_tensor)
                # Assign the convolution value to the corresponding position in the output
                output_tensor[b, c, i // stride] = conv_value

    return output_tensor

# Apply the manual conv1d operation to the input tensor and filter tensor
output_manual = conv1d_manual(input_tensor, filter_tensor, stride=1, padding=0)

# Check if the two tensors match (element-wise comparison)
match = torch.allclose(output_nn, output_manual, atol=1e-6)
print("Both output tensors for conv1d match:", match)

##################################################################################


# Apply the sigmoid activation function using torch.nn.functional
output_nn = torch.nn.Sigmoid()(input_tensor)
def sigmoid_manual(input_tensor):
    # Apply the sigmoid function element-wise
    output_tensor = 1 / (1 + torch.exp(-input_tensor))
    return output_tensor

# Apply the manual sigmoid activation to the input tensor
output_manual = sigmoid_manual(input_tensor)

# Check if the two tensors match (element-wise comparison)
match = torch.allclose(output_nn, output_manual, atol=1e-6)
print("Both output tensors for Sigmoid match:", match)

##################################################################################

# Apply Batch Normalization using torch.nn.BatchNorm1d
bn_layer = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
output_nn = bn_layer(input_tensor)
def batch_norm_manual(input_tensor, eps=1e-05, momentum=0.1):
    # Calculate the mean and variance along the batch dimension
    mean = input_tensor.mean()
    var = input_tensor.var(unbiased=False)
    # Normalize the input tensor using the calculated mean and variance
    normalized_tensor = (input_tensor - mean) / torch.sqrt(var + eps)
    return normalized_tensor

# Apply manual Batch Normalization to the input tensor
output_manual = batch_norm_manual(input_tensor, eps=1e-05, momentum=0.1)
# Check if the two tensors match (element-wise comparison)
match = torch.allclose(output_nn, output_manual, atol=1e-6)
print("Both output tensors for BatchNorm1d match:", match)

##################################################################################

weight = torch.load('4_6_weight.pt')
bias = torch.load('4_6_bias.pt')
# print(weight.shape,bias.shape)
# Define the linear layer
linear_layer = nn.Linear(in_features=32, out_features=16, bias=True)

# Set the weight and bias parameters to the loaded values
linear_layer.weight.data = weight
linear_layer.bias.data = bias

# Apply the linear layer to the input tensor
output_nn = linear_layer(input_tensor)

# Apply Linear Layer manually
def linear_manual(input_tensor, weight, bias):
    # Calculate the linear transformation (matrix multiplication)
    output_manual = input_tensor.matmul(weight.t())

    # Add the bias term
    if bias is not None:
        output_manual += bias

    return output_manual

# Apply manual Linear Layer to the input tensor
output_manual = linear_manual(input_tensor, weight, bias)


# Check if the two tensors match (element-wise comparison)
match = torch.allclose(output_nn, output_manual, atol=1e-6)
print("Both output tensors for Linear match:", match)


