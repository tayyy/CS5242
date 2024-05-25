from torchviz import make_dot
def print_compute_tree(name,node):
    dot = make_dot(node)
    dot.render(name)
import torch
from torchviz import make_dot
from IPython.display import display
import numpy as np



def print_compute_tree(output):
    # Visualize the computation graph and display it in the notebook
    dot = make_dot(output)
    display(dot)

def func_tree(tree , input, k, display=False): 
    input_tensor1 = torch.tensor(input, requires_grad=True)
    input_tensor2 = torch.tensor(input, requires_grad=True)
    input_tensor3 = torch.tensor(input, requires_grad=True)
    input_tensor4 = torch.tensor(input, requires_grad=True)

    if tree == 1:
        layer1 = input_tensor1

        layer2a = layer1 * k
        layer2b = input_tensor2

        layer3_sub = layer2a - layer2b
        layer3_mul = layer2a * layer2b
        layer3_div = layer2b / layer2a

        layer4 = layer3_mul + layer3_div
        output = layer3_sub * layer4

    elif tree == 2:
        layer1_2 = input_tensor2
        layer1_3 = input_tensor3

        layer2_1 = input_tensor1
        layer2_2_add = layer1_2 + k 
        layer2_3_add = layer1_3 + k 

        layer3_1_add = layer2_1 + k 
        layer3_2_mul = layer2_2_add * layer2_3_add

        layer4_1_add = layer3_1_add + layer3_2_mul

        layer5_1_add = layer4_1_add + layer2_2_add

        layer6_1_sub = layer5_1_add - layer3_2_mul
        # layer6_1_sub = layer3_2_mul - layer5_1_add

        layer7_1_add = layer5_1_add + layer6_1_sub
        layer7_2_add = layer6_1_sub + layer2_3_add

        layer8_1_add = layer7_1_add + layer3_2_mul

        layer9_1_mul = layer8_1_add * layer7_2_add

        # layer10_1_div = layer9_1_mul / layer2_3_add
        layer10_1_div = layer2_3_add / layer9_1_mul
        output = layer10_1_div

    elif tree == 3:
        layer1_4 = input_tensor4

        layer2_2 = input_tensor2
        layer2_3 = input_tensor3
        layer2_4_mul = layer1_4 * k

        layer3_2_add = layer2_2 + layer2_3
        layer3_3_add = layer2_3 + k 
        layer3_4_add = layer2_4_mul + k

        layer4_2_sub = layer3_2_add - layer3_3_add
        layer4_3_mul = layer3_3_add * layer3_4_add

        layer5_1 = input_tensor1
        layer5_2_add = layer3_2_add + layer4_2_sub

        layer6_1_add = layer5_1 + layer5_2_add
        layer6_2_add = layer5_1 + layer5_2_add
        layer6_3_sub = layer5_2_add - layer3_3_add

        layer7_1_mul = layer6_2_add * layer6_3_sub
        layer7_2_add = layer4_3_mul + k

        layer8_1_mul = layer7_1_mul * layer6_1_add

        layer8_1_mul = layer8_1_mul / k

        layer10_1_mul = layer8_1_mul * layer7_2_add
        output = layer10_1_mul

    output_numpy = np.round(output.detach().numpy(), decimals=7)
    print(f'output  : {output_numpy}')
    # if display:
    #     print_compute_tree(output)

if __name__ == '__main__':
    constant = 5.0 
    input_train = [1.0, 2.0, 3.0]
    input_question = [4.0, 5.0, 6.0]


    # Question4.1
    output_train = [20.8, 161.6, 542.4]
    print(f'input: {input_train}')
    print(f'expected: {output_train}')
    output = func_tree(tree = 1, input=input_train, k=constant)

    print(f'\ninput: {input_question}')
    output = func_tree(tree = 1, input=input_question, k=constant, display=True)

    # Question4.2
    output_train = [0.0035, 0.0026, 0.0021]
    print(f'input: {input_train}')
    print(f'expected: {output_train}')
    output = func_tree(tree = 2, input=input_train, k=constant)

    print(f'\ninput: {input_question}')
    output = func_tree(tree = 2, input=input_question, k=constant, display=True)

    # Question4.3
    output_train = [-104, -1188, -6468]
    print(f'input: {input_train}')
    print(f'expected: {output_train}')
    output = func_tree(tree = 3, input=input_train, k=constant)

    print(f'\ninput: {input_question}')
    output = func_tree(tree = 3, input=input_question, k=constant, display=True)
