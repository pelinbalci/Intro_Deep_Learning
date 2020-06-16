import torch
from torchvision import datasets, transforms
from torch import nn, optim, topk

from Intro_Pytorch.common import nn_model

# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training & test data:
train_set = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


# Create model with Network
model = nn_model.Network(784, 10, [512, 256, 128])  # input, output, hidden_layer
print('Original model', model)

# Save the model: The parameters of network are stored in a model's state_dict.
torch.save(model.state_dict(), 'checkpoint.pth')

# Load and reuse it:
my_state_dict = torch.load('checkpoint.pth')
model_reuse = nn_model.Network(784, 10, [512, 256, 128])
model_reuse.load_state_dict(my_state_dict)
print('Reuse model:', model_reuse)


# Create another model
model_2 = nn_model.Network(784, 10, [400, 300, 200])
print('Original model_2', model_2)

# Save the model with checkpoint dictionary
checkpoint_2 = {'input_size': 784,
                'output_size': 10,
                'hidden_layers': [each.out_features for each in model_2.hidden_layers],
                'state_dict': model_2.state_dict()}
torch.save(checkpoint_2, 'checkpoint_2.pth')

# Load and reuse the model:
checkpoint_2 = torch.load('checkpoint_2.pth')
model_2_reuse = nn_model.Network(checkpoint_2['input_size'],
                         checkpoint_2['output_size'],
                         checkpoint_2['hidden_layers'])

model_2_reuse.load_state_dict(checkpoint_2['state_dict'])
print('Reuse the model, model_2', model_2_reuse)


# Create another model:
model_3 = nn_model.Network(784, 10, [500, 200, 100])
print('Original model_3', model_3)

# train the model:
criterion = nn.NLLLoss()
optimizer = optim.Adam(model_3.parameters(), lr=0.003)
nn_model.train_model(model_3, train_loader, test_loader, criterion, optimizer, epochs=1, total_steps=40)

# save the model:
torch.save(model_3.state_dict(), 'checkpoint_3.pth')

# Load and reuse the model:
my_state_dict_3 = torch.load('checkpoint_3.pth')
model_reuse_3 = nn_model.Network(784, 10, [500, 200, 100])
model_reuse_3.load_state_dict(my_state_dict_3)
print('Reuse model_3:', model_reuse_3)

# train the model
nn_model.train_model(model_reuse_3, train_loader, test_loader, criterion, optimizer, epochs=1, total_steps=40)

