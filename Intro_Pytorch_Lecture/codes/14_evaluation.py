import torch
from torchvision import datasets, transforms
from torch import nn, optim, topk
import torch.nn.functional as F


# Define a transform to normalize data:
transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ))

# Download and load the training & test data:
trainset = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('fashion_mnist_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# model architecture
class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(784, 256)
        self.hidden_2 = nn.Linear(256, 128)
        self.hidden_3 = nn.Linear(128, 64)
        self.hidden_4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], 784)

        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.log_softmax(self.hidden_4(x), dim=1)

        return x


# 1.define model
# 2. define loss
# 3. define optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# data iteraions
images, labels = next(iter(testloader))

# 4. Forward pass, find the output (model.forward(images))
# 5. Calculate loss with the output and actual values (criterion(logits, labels))
# 6. Clear gradients (optimizer.zero_grad())
# 7. Calculate grad (loss.backward())
# 8. Update weights (optimizer.step())

#logits = model.forward(images)
#loss = criterion(logits, labels)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()


# Get the class probabilities: Check ps.shape = 64 rows,10 columns and ps[i].sum() =1
ps = torch.exp(model(images))


# top probabilities. these are the most likely classes for the first 10 class.
# topk returns k highes value. here k =1. topk(k, dim=1)
# it returns two tensors:
# - top_p --> actual probability values.
# - top_class are the class indices themselves.
top_p, top_class = ps.topk(1, dim=1)
print('probabilities:', top_p[:10, :], 'related classes:', top_class[:10, :])


# top_class.shape = 64,1
# labels.shape = 64
# they need to be in the same shape. we can use view method.
labels.view(*top_class.shape)

# Now we need to check if the predictions match the labels. we can use equals method.
# Equals will have shape 64, 64. It compares one element in each row of top_class in each element in labels.
# It returns 64 boolean values --> True, False
equals = top_class == labels.view(*top_class.shape)

# Example output:
# top_class[1] = 6
# labels.view(*top_class.shape)[1] = 9
# equals = False

# Now we need to calculate the percentage of correct predictions.
# sum the equals / total predictions.
# OR torch.mean(equals) --> it gives error. It only gets floating types, equals' type is boolean.

int(sum(equals)) / equals.shape[0] # 0.0625
sum(equals).type(torch.FloatTensor) / equals.shape[0] #tensor([0.0625])

torch.mean(equals.type(torch.FloatTensor)) #tensor([0.0625])
accuracy = torch.mean(equals.type(torch.FloatTensor)).item() #0.0625


print(accuracy)



