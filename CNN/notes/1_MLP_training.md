Source of all codes and notes is : https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch


## How to code?

    Dataset:
    1. define a transform to normalize data
    2. download data set
    
    Definitions:
    3. define model
    4. define loss function
    6. define optimizer
    
    Training for each epoch & iteration:
    4. Forward pass, find the output (model.forward(images))
    5. Calculate loss with the output and actual values (criterion(logits, labels))
    6. Clear gradients (optimizer.zero_grad())
    7. Calculate grad (loss.backward())  
    8. Update weights (optimizer.step())  


## define a transform to normalize data:
    transform = transforms.Compose((transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ))

## download data set

some datasets are in torchvision --> datasets

parameters are:

    datasets.FashionMNIST('fashion_mnist_data/', download=True, train=True, transform=transform)

If you would like to download test set ---> train=False

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

The batch size is the number of images we get in ONE iteration from the data loader and pass through our network.

In the Mnist dataset, images shape is (64, 1, 28, 28) --> 64 images per batch, 1 color channel, 28x28 images. 

[code](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/4_neural_network.py)

## Iterations

First way:

    for images, labels in trainloader:
        # Flatten Images to 64,784
        images = images.view(images.shape[0], -1)
        
Second way:
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    features = images.view(images.shape[0], 784)    
     

## define model

There are different ways to build a model:


    0. model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))
                      

    1. model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax(dim=1) --> # dim=1 --> calculates softmax across the columns so each row sums to 1.
                      )
                      
     In this model we use LogSoftmax at the end. 
    
    If you use the probabilities to calculate the loss function then you should use NLLLoss. 
    
    While calculating the softmax, remember the architecture of matrices:
    
        There are 64 rows (observations) and 10 columns.
        Each column represent the output for a number. 
        For observation 1 (row), there are 10 outputs. 
        We would like to calculate the  probability of each number
         by dividing exp of one output to sum of exp of each output.
    
    dim=0 takes the sum of the columns. 
    dim=1 takes the sum of the row. 
    
    We didn't use dim argument in nn.Softmax or F.Softmax.
                      
    2.
    class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128,64)
        self.hidden3 = nn.Linear(64,10)

    def forward(self,x):
        #images = images.view(images.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.log_softmax(self.hidden3(x))

        return x
    
                      
     3.
     class Network_alternative(nn.Module):
        def __init__(self):
            super().__init__()
    
            self.hidden = nn.Linear(784, 256)  # Linear Transformation.
            self.output = nn.Linear(256,10)
    
        def forward(self, x):
            # Pass hidden and output layer into sigmoid and softmax functions.
            x = F.sigmoid(self.hidden(x))
            x = F.softmax(self.output(x))
    
            return x
  
[code](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/5_nn_module.py)          

## define loss function

The gradient is the slope of the loss function.

- Forward pass: calculate the loss with given weights.
- Backward pass: calculate the gradient (update the weights) with respect to loss.

Remember softmax? --> e^x1 / e^(x1+x2+..+xn)

What is logsoftmax? --> log(e^x1 / e^(x1+x2+..+xn))

Example:

Output: [2,1,0]
turn these outputs to probability:

    softmax: e^x1 / e^(x1+x2+..+xn):  [0.665, 0.244, 0.090]
    logsoftmax: log(e^x1 / e^(x1+x2+..+xn)): [-0.407, -1.407, -2.407]
    exp(logsoftmax): exp(log(e^x1 / e^(x1+x2+..+xn))): [0.665, 0.244, 0.090]

--

    cross entropy:  Sum of the negative of the logarithm of probabilities
    cross entropy: - Sum over(i = 1..m) yi * ln(pi) + (1-yi) * ln(1-pi)


- If you use Logsoftmax for the last layer --> nn.NLLLoss()` --> negative log likelihood loss.
- If you don't use Logsoftmax for the last layer --> `nn.CrossEntropyLoss()` --> pass the raw output into the loss.

[code](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/7_train_neural_network.py)


## backpropagation

Autograd automatically calculates the gradients.
Turn on or off gradients altogether with `torch.set_grad_enabled(True|False)`

[code](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/8_backpropagation.py)

## define optimizer

Training the network. 
How can we update the weights with gradients? We need to define this function. 
Pytorch has optim package. 
Stochastic gradient descent --> `optim.SGD`

When you do multiple backwards with the same parameters the gradients are accumulated.
You need to zero the gradients on each training pass.
Otherwise, you'll retain gradients from previous training batches.

Clear the gradients:

    optimizer.zero_grad()

[code](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/10_train_neural_network.py)


## Summary

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden1 = nn.Linear(784, 128)
            self.hidden2 = nn.Linear(128,64)
            self.hidden3 = nn.Linear(64,10)
    
        def forward(self,x):
            #images = images.view(images.shape[0], -1)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.log_softmax(self.hidden3(x))
    
            return x
    
    # Define model:
    model = Classifier()
    
    # Define Loss:
    criterion = nn.NLLLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 5
    for e in range(epochs):
        epoch_loss = 0
        for images, labels in trainloader:
    
            # Forward pass, find the output
            logits = model.forward(images)
    
            # Calculate loss with the output and actual values
            loss = criterion(logits, labels)
    
            # Clear the gradient
            optimizer.zero_grad()
            
            # Calculate grad
            loss.backward()  
            
            # Update weights
            optimizer.step()  
    
            epoch_loss += loss.item()
 
 [code](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/13_classify_fashion.py)