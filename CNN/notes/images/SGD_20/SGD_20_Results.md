/Users/pelin.balci/PycharmProjects/aws_machine_learning/Intro_Pytorch/CNN/codes/Mnist_mlp.py

Model:

    Network(
      (fc1): Linear(in_features=784, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=256, bias=True)
      (fc3): Linear(in_features=256, out_features=10, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )

Training Losses:

    epoch: 1 \TrainingLoss: 17.287962
    epoch: 2 \TrainingLoss: 6.660665
    epoch: 3 \TrainingLoss: 5.159274
    epoch: 4 \TrainingLoss: 4.194034
    epoch: 5 \TrainingLoss: 3.507730
    epoch: 6 \TrainingLoss: 3.003063
    epoch: 7 \TrainingLoss: 2.668572
    epoch: 8 \TrainingLoss: 2.365989
    epoch: 9 \TrainingLoss: 2.139566
    epoch: 10 \TrainingLoss: 1.942329
    epoch: 11 \TrainingLoss: 1.784263
    epoch: 12 \TrainingLoss: 1.639998
    epoch: 13 \TrainingLoss: 1.501188
    epoch: 14 \TrainingLoss: 1.418957
    epoch: 15 \TrainingLoss: 1.331933
    epoch: 16 \TrainingLoss: 1.231623
    epoch: 17 \TrainingLoss: 1.169610
    epoch: 18 \TrainingLoss: 1.078209
    epoch: 19 \TrainingLoss: 1.046653
    epoch: 20 \TrainingLoss: 0.987240
    
Model = SGD; Total time: 118.666 seconds

SGD for 20 epochs,  training loss: 0.9872399334182652

Test Loss: 0.063526

    Test Accuracy of     0: 98% (970/980)
    Test Accuracy of     1: 99% (1127/1135)
    Test Accuracy of     2: 97% (1007/1032)
    Test Accuracy of     3: 98% (990/1010)
    Test Accuracy of     4: 98% (967/982)
    Test Accuracy of     5: 98% (877/892)
    Test Accuracy of     6: 98% (939/958)
    Test Accuracy of     7: 97% (1003/1028)
    Test Accuracy of     8: 96% (942/974)
    Test Accuracy of     9: 97% (981/1009)

Test Accuracy (Overall): 98% (9803/10000)