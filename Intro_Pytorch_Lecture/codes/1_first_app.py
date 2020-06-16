import torch


def activation(x):
    return 1/(1+torch.exp(-x))


# generate some data:
torch.manual_seed(7)

# randn produce normally distributed data. 1 row & 5 columns
features = torch.randn((1, 5))  # tensor([[-0.1468,  0.7861,  0.9468, -1.1143,  1.6908]])

# randn_like creates data with the same shape as features
weights = torch.randn_like(features)  # tensor([[-0.8948, -0.3556,  1.2324,  0.1382, -1.6822]])

# randn produce normally distributed data. 1 row & 1 column
bias = torch.randn((1, 1))  # tensor([[0.3177]])


print('output calculation')

y_sum = activation((features * weights).sum() + bias)
y_torch_sum = activation(torch.sum(features * weights) + bias)

# use mm instead of matmul
y_mm = activation(torch.mm(weights, features.T) + bias)
y_matmul = activation(torch.matmul(weights, features.T) + bias)

# You can use tensor.shape
# features.reshape(5, 1) --> copies data to another memory.
# feature.resize_(5, 1) --> inplace operation. same tensor to different shape.
# feature.view(5, 1) --> a new tensor with same data.

# use view.
y = activation(torch.mm(features, weights.view(5, 1)) + bias)

print(y_sum)
print(y_torch_sum)
print(y_mm)
print(y_matmul)
print(y)

