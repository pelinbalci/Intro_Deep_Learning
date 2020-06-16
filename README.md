# Intro_Deep_Learning

I'm taking "Introduction to Deep Learning with PyTorch" lecture from Udacity. 
This repository includes my notes & codes from this lecture. I also take screenshots to remember the concepts easily.
There are also some useful links and my own notes. 

I would like to thank Udacity to give us oppotunity to reach this course free:)

- The course link: https://classroom.udacity.com/courses/ud188
- The original codes: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch

The outline of the repository

1. [Intro_NN](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN) 
2. [Intro_Pytorch_Lecture](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_Pytorch_Lecture) 
3. [CNN](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/CNN) 
4. [common](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/common) 

The details:

1. [Intro_NN](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN) 

Introduction to Neural Networks. Before using the torch modules it is useful to understand the maths behind gradients, perceptron, regularization. I also add some of my handwritten notes not to forget the derivatives :)

- [Notes](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/notes): 
  - [perceptron_math](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/notes/1_Perceptron_math.md),
  - [cross_entropy](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/notes/2_Cross_Entropy.md)
  - [gradients](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/notes/3_Gradient.md), 
  - [neural_network](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/notes/4_Neural_Network.md), 
  - [overfitting](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/notes/5_Overfitting.md),
  - [other_problems](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/notes/6_Other_Problems.md)
  
- [Codes](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes): small scripts about this topics.
  - [perceptrons.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/Perceptrons.py)
  - [cross_entropy.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/cross_entropy.py)
  - [gradient_descent.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/gradient_descent.py)
  - [gradient_lin_reg.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/gradient_lin_reg.py)
  - [perceptron_algorithm.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/perceptron_algorithm.py)
  - [softmax.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/softmax.py)
  - [loss_explanation.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/loss_explanation.py)
  - [neural_network_example.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_NN/intro_codes/student_data.py)
  

2. [Intro_Pytorch_Lecture](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_Pytorch_Lecture) 
Learn how to develeop a neural network with PyTorch.
Mnist Data Set, Mnist Fashion Data Set and Cat & Dog images are used for classification. 

- [check_pytorch.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/0_check_pytorch.py)
- [simple_examples.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/1_first_app.py)
- [define_multilayer.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/2_multilayer.py)
- [numpy_to_tensor.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/3_numpy_to_tensor.py) --> How to turn numpy array to tensor?
- [define_nn.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/4_neural_network.py) --> First neural network of Mnist data with torch
- [define_nn_module.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/5_nn_module.py) --> It has two types of module; first one uses torch.nn module, second uses torch.nn.functional module
- [logsoftmax.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/5_1_logsoftmax.py) --> The output of logsoftmax function
- [apply_nn_module.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/6_neural_network_extended.py) --> Application of nn module to Mnist data. 
- [train_nn](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/7_train_neural_network.py) --> Train the meural network and find the loss
- [backpropagation.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/8_backpropagation.py) --> Simple example of backpropagation
- [apply_backward.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/9_apply_backward.py) --> Application of backward to Mnist data. 
- [weight_changes.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/10_train_neural_network.py) --> shows how the weights change after applying backward. 
- [train_mnist.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/11_train_neural_network.py) -->  shows training loop. Nice plot for the prediction at the end.
- [train_mnist_fashion.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/12_classify_fashion.py) --> Tranining loop for fashion data. 
- [train_mnist_fashion_2.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/13_classify_fashion.py) --> Tranining loop for fashion data: the optimization function is Adam.
- [recap_training.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/13_1_recap.py) --> I simplified the code to understand better:)
- [validation.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/14_validation.py) --> How can we use validation? There are lots of explanations about validation process. 
- [apply_validation.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/15_apply_validation.py) --> Application of validation process to fashion data. 
- [prevent_overfitting.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/16_prevent_overfitting.py) -->  It is the first time we use dropouts. 
- [save_load_models_1.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/17_save_load_models.py) --> includes notes and outputs. 
- [save_load_models_2.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/17_1_save_models.py) --> This is better to understand that saving and loading process. 
- [load_cat_dog_data](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/18_load_cat_dog_data.py) -->  I load the data and try to apply the training and validation process. The right way is using transfer learning. 
- [transfer_learning_1.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/19_transfer_learning_1.py) -->  includes densenet121 details. Also you can compare the cpu and gpu times. However I could not use gpu on my computer:(
- [apply_transfer_learning](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_Pytorch_Lecture/codes/20_transfer_learning_2.py) -->  Since I'm using cpu, the code takes forever. 
- [full_transfer_learning.py](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/Intro_Pytorch_Lecture/codes) --> Final code for transfer learning. 

3. [CNN](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/CNN) 
The first part is a recap for MLP. You can find complete classification code for Mnist dataset and analysis of these codes.

- [Notes](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/CNN/notes):
  - [training_process](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/CNN/notes/1_MLP_training.md) --> Explanations for training part. This file explains the code of the nn model in detail. 
  - [summary](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/CNN/notes/2_MLP_Recap.md) --> Summary for Neural Networks. 
  - [Resources](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/CNN/notes/0_Resources_for_CNN.md) --> Useful links for CNN.

- [Codes](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/CNN/codes):
  - [Mnist_data_trainin.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/CNN/codes/1_Mnist_mlp.py) --> Full code for mnist data training. I made some analysis for different optim models here. 
  - [Mnist_data_validation.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/CNN/codes/2_Mnist_MLP_validation.py) --> How can we add validation set? And what is sampler? Check [this](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/CNN/codes/Sampler.py)


4. [common](https://github.com/pelinbalci/Intro_Deep_Learning/tree/master/common) 
This folder includes common scripts which are used for the entire project. 
- [helper.py](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/common/helper.py) --> is provided by Udacity, it is useful for plots. 
- [fc_model](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/common/fc_model.py) and [nn_model](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/common/nn_model.py) --> are almost the same, reprsents a basic neural network. 
- [save_fig](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/common/save_fig.py) --> is used for saving the plots. 
