import torch
from torchvision import datasets, transforms
from torch import nn, optim, topk
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], 784)

        x = self.dropout(F.relu(self.hidden_1(x)))
        x = self.dropout(F.relu(self.hidden_2(x)))
        x = self.dropout(F.relu(self.hidden_3(x)))

        # no dropout for the final layer. It should have 10 neurons.
        x = F.log_softmax(self.hidden_4(x), dim=1)

        return x


# define model, loss, optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# train & validate model
epochs = 30
train_losses, test_losses = [],[]

for epoch in range(epochs):
    train_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        logits = model.forward(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    else:  # after the first loop completes, this part will run

        test_loss = 0
        accuracy = 0

        with torch.no_grad():  # we won't do any backward operation for the validation part.

            # set model to evaluation mode
            model.eval()  # This sets the model to evaluation mode where the dropout probability is 0.

            for images, labels in testloader:
                # Get the output
                logits_test = model(images)

                # Calculate loss
                test_loss += criterion(logits_test, labels)

                # probability of the output
                ps = torch.exp(logits_test)

                # top probabilities and classes
                top_p, top_class = ps.topk(1, dim=1)

                # compare the predictions and labels
                equals = top_class == labels.view(*top_class.shape)

                # calculate accuracy
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train() # back to training mode

        test_losses.append(test_loss/len(testloader))
        train_losses.append(train_loss/len(trainloader))

        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

plt.plot(test_losses, label='test')
plt.plot(train_losses, label='train')
plt.legend()
plt.show()

'''
Epoch: 1/30..  Training Loss: 0.605..  Test Loss: 0.506..  Test Accuracy: 0.818
Epoch: 2/30..  Training Loss: 0.487..  Test Loss: 0.455..  Test Accuracy: 0.840
Epoch: 3/30..  Training Loss: 0.451..  Test Loss: 0.414..  Test Accuracy: 0.850
Epoch: 4/30..  Training Loss: 0.433..  Test Loss: 0.412..  Test Accuracy: 0.853
Epoch: 5/30..  Training Loss: 0.423..  Test Loss: 0.415..  Test Accuracy: 0.850
Epoch: 6/30..  Training Loss: 0.407..  Test Loss: 0.405..  Test Accuracy: 0.860
Epoch: 7/30..  Training Loss: 0.410..  Test Loss: 0.415..  Test Accuracy: 0.849
Epoch: 8/30..  Training Loss: 0.399..  Test Loss: 0.408..  Test Accuracy: 0.851
Epoch: 9/30..  Training Loss: 0.400..  Test Loss: 0.391..  Test Accuracy: 0.860
Epoch: 10/30..  Training Loss: 0.387..  Test Loss: 0.386..  Test Accuracy: 0.863
Epoch: 11/30..  Training Loss: 0.383..  Test Loss: 0.392..  Test Accuracy: 0.865
Epoch: 12/30..  Training Loss: 0.381..  Test Loss: 0.392..  Test Accuracy: 0.862
Epoch: 13/30..  Training Loss: 0.378..  Test Loss: 0.387..  Test Accuracy: 0.867
Epoch: 14/30..  Training Loss: 0.375..  Test Loss: 0.393..  Test Accuracy: 0.862
Epoch: 15/30..  Training Loss: 0.375..  Test Loss: 0.386..  Test Accuracy: 0.866
Epoch: 16/30..  Training Loss: 0.368..  Test Loss: 0.374..  Test Accuracy: 0.871
Epoch: 17/30..  Training Loss: 0.365..  Test Loss: 0.405..  Test Accuracy: 0.860
Epoch: 18/30..  Training Loss: 0.372..  Test Loss: 0.384..  Test Accuracy: 0.867
Epoch: 19/30..  Training Loss: 0.365..  Test Loss: 0.390..  Test Accuracy: 0.868
Epoch: 20/30..  Training Loss: 0.356..  Test Loss: 0.377..  Test Accuracy: 0.876
Epoch: 21/30..  Training Loss: 0.358..  Test Loss: 0.418..  Test Accuracy: 0.859
Epoch: 22/30..  Training Loss: 0.357..  Test Loss: 0.378..  Test Accuracy: 0.871
Epoch: 23/30..  Training Loss: 0.363..  Test Loss: 0.379..  Test Accuracy: 0.865
Epoch: 24/30..  Training Loss: 0.353..  Test Loss: 0.383..  Test Accuracy: 0.872
Epoch: 25/30..  Training Loss: 0.358..  Test Loss: 0.380..  Test Accuracy: 0.872
Epoch: 26/30..  Training Loss: 0.354..  Test Loss: 0.393..  Test Accuracy: 0.866
Epoch: 27/30..  Training Loss: 0.344..  Test Loss: 0.371..  Test Accuracy: 0.876
Epoch: 28/30..  Training Loss: 0.346..  Test Loss: 0.378..  Test Accuracy: 0.875
Epoch: 29/30..  Training Loss: 0.346..  Test Loss: 0.387..  Test Accuracy: 0.866
Epoch: 30/30..  Training Loss: 0.346..  Test Loss: 0.378..  Test Accuracy: 0.874

'''
