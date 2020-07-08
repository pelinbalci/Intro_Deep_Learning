/Users/pelin.balci/PycharmProjects/aws_machine_learning/Intro_Pytorch/CNN/codes/Mnist_mlp.py

Model:

    Network(
      (fc1): Linear(in_features=784, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=256, bias=True)
      (fc3): Linear(in_features=256, out_features=10, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
    
Training Losses:

    epoch: 1 \TrainingLoss: 11.077646
    epoch: 2 \TrainingLoss: 9.590197
    epoch: 3 \TrainingLoss: 9.073283
    epoch: 4 \TrainingLoss: 8.946371
    epoch: 5 \TrainingLoss: 8.442004
    epoch: 6 \TrainingLoss: 8.343152
    epoch: 7 \TrainingLoss: 8.467776
    epoch: 8 \TrainingLoss: 7.783962
    epoch: 9 \TrainingLoss: 8.587967
    epoch: 10 \TrainingLoss: 8.564115
    epoch: 11 \TrainingLoss: 8.275888
    epoch: 12 \TrainingLoss: 8.621009
    epoch: 13 \TrainingLoss: 8.846032
    epoch: 14 \TrainingLoss: 8.411981
    epoch: 15 \TrainingLoss: 8.525838
    epoch: 16 \TrainingLoss: 8.357536
    epoch: 17 \TrainingLoss: 8.595840
    epoch: 18 \TrainingLoss: 8.447555
    epoch: 19 \TrainingLoss: 9.159551
    epoch: 20 \TrainingLoss: 8.822327

Model = Adam; Total time: 322.621 seconds

Adam for 20 epochs,  training loss: 8.822327231214459

Test Loss: 0.420031

    Test Accuracy of     0: 94% (922/980)
    Test Accuracy of     1: 95% (1079/1135)
    Test Accuracy of     2: 88% (909/1032)
    Test Accuracy of     3: 91% (924/1010)
    Test Accuracy of     4: 95% (933/982)
    Test Accuracy of     5: 95% (854/892)
    Test Accuracy of     6: 96% (921/958)
    Test Accuracy of     7: 91% (942/1028)
    Test Accuracy of     8: 91% (889/974)
    Test Accuracy of     9: 90% (913/1009)

Test Accuracy (Overall): 92% (9286/10000)

Process finished with exit code 0