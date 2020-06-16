#https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html

import torch
from torch import nn

softmax = nn.Softmax()
logsoftmax = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()

output_lin_function = torch.randn(2, 2, requires_grad=True)
target = torch.tensor([1, 0])

softmax_output = softmax(output_lin_function)
logsoftmax_output = logsoftmax(output_lin_function)

calculate_loss = loss(logsoftmax_output, target)

'''
input
tensor([[1.8902, 1.2803],
        [1.1482, 0.1605]], requires_grad=True)

softmax:
tensor([[0.6479, 0.3521],  --> sum = 1
        [0.7286, 0.2714]], grad_fn=<SoftmaxBackward>)  --> sum = 1

log_prob_input
tensor([[-0.4340, -1.0439],
        [-0.3166, -1.3042]], grad_fn=<LogSoftmaxBackward>)
        
target
tensor([1, 0])

loss       
tensor(0.6802, grad_fn=<NllLossBackward>)
'''

softmax = nn.Softmax()
logsoftmax = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
cross_entropy_loss = nn.CrossEntropyLoss()

output_lin_function = torch.randn(3, 3, requires_grad=True)
target = torch.tensor([2, 1, 0])

softmax_output = softmax(output_lin_function)
logsoftmax_output = logsoftmax(output_lin_function)

calculate_loss = loss(logsoftmax_output, target)
calculate_loss_2 = cross_entropy_loss(output_lin_function, target)

'''
input
tensor([[ 0.0195,  0.0408, -1.5863],
        [ 0.5292, -0.4793,  0.9122],
        [ 0.3554,  0.2140,  0.7718]], requires_grad=True)

softmax_output:
tensor([[0.4500, 0.4597, 0.0903],
        [0.3532, 0.1288, 0.5180],
        [0.2955, 0.2565, 0.4480]], grad_fn=<SoftmaxBackward>)

log_softmax_output
tensor([[-0.7985, -0.7772, -2.4043],
        [-1.0408, -2.0493, -0.6578],
        [-1.2193, -1.3606, -0.8029]], grad_fn=<LogSoftmaxBackward>)

target
tensor([2, 1, 0])

loss       
tensor(1.8910, grad_fn=<NllLossBackward>)
same as cross entropy loss. =)
'''



print('done')
