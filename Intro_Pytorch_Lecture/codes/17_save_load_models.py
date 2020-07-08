import torch
from torchvision import datasets, transforms
from torch import nn, optim, topk
from common import nn_model


# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training & test data:
train_set = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


# define model, loss, optimizer
# model = nn_model.Network() --> it doesn't work. We need to define the units.

model = nn_model.Network(784, 10, [512, 256, 128])  # input, output, hidden_layer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

#nn_model.train_model(model, train_loader, test_loader, criterion, optimizer, epochs=1, total_steps=40)


# Save the model: The parameters of network are stored in a model's state_dict.
print('model:', model, '/n')
print('the state dict keys: ', model.state_dict().keys())

'''
model: Classifier(
  (hidden_1): Linear(in_features=784, out_features=256, bias=True)
  (hidden_2): Linear(in_features=256, out_features=128, bias=True)
  (hidden_3): Linear(in_features=128, out_features=64, bias=True)
  (hidden_4): Linear(in_features=64, out_features=10, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
) /n
the state dict keys:  odict_keys(['hidden_1.weight', 'hidden_1.bias', 'hidden_2.weight', 'hidden_2.bias', 'hidden_3.weight', 'hidden_3.bias', 'hidden_4.weight', 'hidden_4.bias'])

'''

torch.save(model.state_dict(), 'checkpoint.pth')


# Load the model
my_state_dict = torch.load('checkpoint.pth')
print(my_state_dict)

'''
keys: the state dict keys:  odict_keys(['hidden_1.weight', 'hidden_1.bias', 'hidden_2.weight', 'hidden_2.bias', 
'hidden_3.weight', 'hidden_3.bias', 'hidden_4.weight', 'hidden_4.bias'])

ordered dict: 
([('hidden_1.weight', tensor([[ 0.0066, -0.0236, -0.0331,  ...,  0.0303, -0.0152,  0.0146],
                              [ 0.0195, -0.0091,  0.0308,  ...,  0.0254,  0.0195, -0.0228],
                              [ 0.0318, -0.0106,  0.0344,  ...,  0.0149, -0.0039, -0.0274],
                               ...,
                              [ 0.0222,  0.0183,  0.0176,  ..., -0.0167,  0.0252, -0.0320],
                              [ 0.0346, -0.0204,  0.0354,  ...,  0.0173, -0.0133, -0.0075],
                              [ 0.0177,  0.0343,  0.0022,  ..., -0.0189,  0.0088, -0.0119]])), 
('hidden_1.bias', tensor([ 2.2183e-02,  2.3394e-02, ...]) '''

model.load_state_dict(my_state_dict)

# If you want to change the model architecture:
model = nn_model.Network(784, 10, [400, 300, 120])
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
checkpoint = torch.load('checkpoint.pth')
model = nn_model.Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'])
model.load_state_dict(checkpoint['state_dict'])

print('model')
