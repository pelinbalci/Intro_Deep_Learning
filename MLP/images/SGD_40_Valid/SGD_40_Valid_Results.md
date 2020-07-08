/Users/pelin.balci/PycharmProjects/aws_machine_learning/Intro_Pytorch/CNN/codes/2_Mnist_MLP_validation.py

Model: 

    Net(
      (fc1): Linear(in_features=784, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=512, bias=True)
      (fc3): Linear(in_features=512, out_features=10, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )

Training and Validation Losses:

    Epoch: 1 	Training Loss: 0.940147 	Validation Loss: 0.369444
    Validation loss decreased (inf --> 0.369444).  Saving model ...
    Epoch: 2 	Training Loss: 0.356912 	Validation Loss: 0.289844
    Validation loss decreased (0.369444 --> 0.289844).  Saving model ...
    Epoch: 3 	Training Loss: 0.282618 	Validation Loss: 0.237172
    Validation loss decreased (0.289844 --> 0.237172).  Saving model ...
    Epoch: 4 	Training Loss: 0.232976 	Validation Loss: 0.206138
    Validation loss decreased (0.237172 --> 0.206138).  Saving model ...
    Epoch: 5 	Training Loss: 0.197966 	Validation Loss: 0.180391
    Validation loss decreased (0.206138 --> 0.180391).  Saving model ...
    Epoch: 6 	Training Loss: 0.171593 	Validation Loss: 0.160178
    Validation loss decreased (0.180391 --> 0.160178).  Saving model ...
    Epoch: 7 	Training Loss: 0.152824 	Validation Loss: 0.145555
    Validation loss decreased (0.160178 --> 0.145555).  Saving model ...
    Epoch: 8 	Training Loss: 0.134486 	Validation Loss: 0.135806
    Validation loss decreased (0.145555 --> 0.135806).  Saving model ...
    Epoch: 9 	Training Loss: 0.121796 	Validation Loss: 0.130589
    Validation loss decreased (0.135806 --> 0.130589).  Saving model ...
    Epoch: 10 	Training Loss: 0.111758 	Validation Loss: 0.120006
    Validation loss decreased (0.130589 --> 0.120006).  Saving model ...
    Epoch: 11 	Training Loss: 0.100249 	Validation Loss: 0.112469
    Validation loss decreased (0.120006 --> 0.112469).  Saving model ...
    Epoch: 12 	Training Loss: 0.093399 	Validation Loss: 0.104627
    Validation loss decreased (0.112469 --> 0.104627).  Saving model ...
    Epoch: 13 	Training Loss: 0.085587 	Validation Loss: 0.102818
    Validation loss decreased (0.104627 --> 0.102818).  Saving model ...
    Epoch: 14 	Training Loss: 0.078427 	Validation Loss: 0.096803
    Validation loss decreased (0.102818 --> 0.096803).  Saving model ...
    Epoch: 15 	Training Loss: 0.073807 	Validation Loss: 0.094912
    Validation loss decreased (0.096803 --> 0.094912).  Saving model ...
    Epoch: 16 	Training Loss: 0.069523 	Validation Loss: 0.091361
    Validation loss decreased (0.094912 --> 0.091361).  Saving model ...
    Epoch: 17 	Training Loss: 0.064011 	Validation Loss: 0.088503
    Validation loss decreased (0.091361 --> 0.088503).  Saving model ...
    Epoch: 18 	Training Loss: 0.060058 	Validation Loss: 0.085506
    Validation loss decreased (0.088503 --> 0.085506).  Saving model ...
    Epoch: 19 	Training Loss: 0.057192 	Validation Loss: 0.086481
    Epoch: 20 	Training Loss: 0.052738 	Validation Loss: 0.083803
    Validation loss decreased (0.085506 --> 0.083803).  Saving model ...
    Epoch: 21 	Training Loss: 0.049713 	Validation Loss: 0.082091
    Validation loss decreased (0.083803 --> 0.082091).  Saving model ...
    Epoch: 22 	Training Loss: 0.047279 	Validation Loss: 0.080808
    Validation loss decreased (0.082091 --> 0.080808).  Saving model ...
    Epoch: 23 	Training Loss: 0.043390 	Validation Loss: 0.080850
    Epoch: 24 	Training Loss: 0.040692 	Validation Loss: 0.078559
    Validation loss decreased (0.080808 --> 0.078559).  Saving model ...
    Epoch: 25 	Training Loss: 0.039993 	Validation Loss: 0.077313
    Validation loss decreased (0.078559 --> 0.077313).  Saving model ...
    Epoch: 26 	Training Loss: 0.036056 	Validation Loss: 0.076329
    Validation loss decreased (0.077313 --> 0.076329).  Saving model ...
    Epoch: 27 	Training Loss: 0.034684 	Validation Loss: 0.080086
    Epoch: 28 	Training Loss: 0.033650 	Validation Loss: 0.074381
    Validation loss decreased (0.076329 --> 0.074381).  Saving model ...
    Epoch: 29 	Training Loss: 0.032044 	Validation Loss: 0.076104
    Epoch: 30 	Training Loss: 0.029888 	Validation Loss: 0.074260
    Validation loss decreased (0.074381 --> 0.074260).  Saving model ...
    Epoch: 31 	Training Loss: 0.028517 	Validation Loss: 0.076512
    Epoch: 32 	Training Loss: 0.026353 	Validation Loss: 0.074133
    Validation loss decreased (0.074260 --> 0.074133).  Saving model ...
    Epoch: 33 	Training Loss: 0.025431 	Validation Loss: 0.073304
    Validation loss decreased (0.074133 --> 0.073304).  Saving model ...
    Epoch: 34 	Training Loss: 0.025197 	Validation Loss: 0.073359
    Epoch: 35 	Training Loss: 0.023447 	Validation Loss: 0.073918
    Epoch: 36 	Training Loss: 0.022193 	Validation Loss: 0.073045
    Validation loss decreased (0.073304 --> 0.073045).  Saving model ...
    Epoch: 37 	Training Loss: 0.021852 	Validation Loss: 0.071440
    Validation loss decreased (0.073045 --> 0.071440).  Saving model ...
    Epoch: 38 	Training Loss: 0.020700 	Validation Loss: 0.074503
    Epoch: 39 	Training Loss: 0.018578 	Validation Loss: 0.072838
    Epoch: 40 	Training Loss: 0.018308 	Validation Loss: 0.072382
    
Test Loss: 0.057664


Test Accuracy: 

    Test Accuracy of     0: 99% (971/980)
    Test Accuracy of     1: 99% (1126/1135)
    Test Accuracy of     2: 98% (1012/1032)
    Test Accuracy of     3: 98% (992/1010)
    Test Accuracy of     4: 97% (959/982)
    Test Accuracy of     5: 98% (876/892)
    Test Accuracy of     6: 98% (940/958)
    Test Accuracy of     7: 97% (998/1028)
    Test Accuracy of     8: 97% (951/974)
    Test Accuracy of     9: 97% (988/1009)

Test Accuracy (Overall): 98% (9813/10000)