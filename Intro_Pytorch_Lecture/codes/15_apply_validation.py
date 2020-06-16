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

    def forward(self, x):
        x = x.view(x.shape[0], 784)

        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.log_softmax(self.hidden_4(x), dim=1)

        return x


# define model, loss, optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# train & validate model
epochs = 3
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

        test_losses.append(test_loss/len(testloader))
        train_losses.append(train_loss/len(trainloader))

        print(f'training loss: {train_loss/len(trainloader)}')
        print(f'test loss: {test_loss/len(testloader)}')
        print(f'test accuracy: {accuracy/len(testloader)}')

        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

plt.plot(test_losses, label='test')
plt.plot(train_losses, label='train')
plt.legend()
plt.show()

'''

training loss: 0.5176134678219427
test loss: 0.4426700472831726
test accuracy: 0.8436504602432251
training loss: 0.3917560773585905
test loss: 0.40145790576934814
test accuracy: 0.8583797812461853
training loss: 0.3588648996214623
test loss: 0.389749139547348
test accuracy: 0.8584793210029602
training loss: 0.334318279298638
test loss: 0.3863457143306732
test accuracy: 0.8675358295440674
training loss: 0.3165982878490933
test loss: 0.35813990235328674
test accuracy: 0.868829607963562
training loss: 0.30487688485858666
test loss: 0.3815162777900696
test accuracy: 0.8664410710334778
training loss: 0.29303953679068
test loss: 0.4045107066631317
test accuracy: 0.8677348494529724
training loss: 0.28395184294691983
test loss: 0.36629244685173035
test accuracy: 0.8759952187538147
training loss: 0.280083072830492
test loss: 0.3669793903827667
test accuracy: 0.8723129034042358
training loss: 0.267419929459278
test loss: 0.3696724474430084
test accuracy: 0.8701233863830566
training loss: 0.2594739778447888
test loss: 0.3622663617134094
test accuracy: 0.8799760937690735
training loss: 0.25305513321940326
test loss: 0.4014567732810974
test accuracy: 0.8749004602432251
training loss: 0.25221487253046493
test loss: 0.3781089186668396
test accuracy: 0.8762937784194946
training loss: 0.2437027535204694
test loss: 0.36731889843940735
test accuracy: 0.877886176109314
training loss: 0.23805485469207707
test loss: 0.4114396870136261
test accuracy: 0.875199019908905
training loss: 0.2332354588295097
test loss: 0.3732375502586365
test accuracy: 0.8800756335258484
training loss: 0.22724514601152462
test loss: 0.3818086087703705
test accuracy: 0.8821656107902527
training loss: 0.2300220605339418
test loss: 0.36590734124183655
test accuracy: 0.8769904375076294
training loss: 0.22031581449086096
test loss: 0.3992060422897339
test accuracy: 0.8800756335258484
training loss: 0.22530208468071813
test loss: 0.4060320258140564
test accuracy: 0.878284215927124
training loss: 0.21650883926352713
test loss: 0.37986454367637634
test accuracy: 0.8836584687232971
training loss: 0.2098601703633314
test loss: 0.39857903122901917
test accuracy: 0.8835589289665222
training loss: 0.21056178137104012
test loss: 0.40786921977996826
test accuracy: 0.8815684914588928
training loss: 0.20670605050936056
test loss: 0.39799752831459045
test accuracy: 0.8749004602432251
training loss: 0.2012048180439452
test loss: 0.3937186598777771
test accuracy: 0.8834593892097473
training loss: 0.19986984873218322
test loss: 0.4202917814254761
test accuracy: 0.8796775341033936
training loss: 0.19460011985121187
test loss: 0.411224901676178
test accuracy: 0.8774880766868591
training loss: 0.19196091698748724
test loss: 0.4193893373012543
test accuracy: 0.8804737329483032
training loss: 0.20101231173363956
test loss: 0.40105509757995605
test accuracy: 0.883957028388977
training loss: 0.1959881079611557
test loss: 0.4092176854610443
test accuracy: 0.8836584687232971
'''