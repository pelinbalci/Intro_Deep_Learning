
Example-1:
A linear model is a probability space. 

It gives us the probability of being blue (class 1, positive class) for every point.

In one prob. space a point has a prob = 0.8 of being blue. 
In second prob. space ıt has a prob of being blue = 0.7 

How can we combine them?

Summarize? 

0.8 + 0.7 = 1.5 :( It's not btw 0-1 . 

    We can use sigmoid function --> sigmoid(p1 + p2)

sigmoid(0.8 + 0.7) = 0.82

    we can weight the probabilities ---> sigmoid(w1*p1 + w2*p2 + b)
    
sigmoid(7 * 0.7 + 5 * 0.8 -6) = sigmoid(2.9) = 0.95


Example-2:

- 1st linear model = 5x1 - 2x2 + 8 
- 2nd linear model = 7x1 - 3x2 + 1

from the first linear model we need to calculate a probability, so we need to use sigmoid function for it. 

- sigmoid(5x1 - 2x2 + 8)) --> prob of first model for the point x1, x2 --> a
- sigmoid(7x1 - 3x2 + 1)) --> prob of second model for the point x1, x2 ---> b

- Combine the outputs with the weights; 7 and 5 and use bias as -6.

we can use a and b directly. 

if the combination model is 7a + 5b -6, we need to use sigmoid again;

sigmoid(7a + 5b -6) ---> output probability. 

![image_description](images/nn_architecture.png) 

this image shows that:

- the first layer is input layer. 
- the second layer is hidden layer. There are activation functions like sigmoid.
- the third layer is output layer. There is another activation function and we find the final probability. 

## Multiclass classification

We want a model that shows us an image is a dog, a cat or a bear. 

We don't need to create 3 different models. We just need to add 3 activation functions at the output layer. 

Question: How many nodes in the output layer would you require if you were trying to classify all the letters in the English alphabet?
Answer: 26

## Feedforward

Feedforward is the process neural networks use to turn the input into an output

y' = &sigma; * W(2) *  &sigma; * W(1) * x


## Error Function

![image_description!](images/nn_error.png)

## Backpropagation

Ref: http://www.ashukumar27.io/LogisticRegression-Backpropagation/
Ref: https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
Ref: https://mc.ai/derivation-of-back-propagation-with-cross-entropy/
Ref: https://medium.com/@pchetan481/derivation-of-back-propagation-with-cross-entropy-7ac0f63c8c28

- Doing a feedforward operation.
- Comparing the output of the model with the desired output.
- Calculating the error.
- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
- Use this to update the weights, and get a better model.
- Continue this until we have a model that is good. 

![image_description!](images/backprop.jpeg)


## Summary

![image_description!](images/summary_1.jpeg)

![image_description!](images/summary_2.jpeg)


# Summary:

Our problem is predicting the images. We have two images: dog, cat --> there are 2 classes. And we have 10 observations. 

    y = [0 1 1.... 0
         1 0 0.....1
     
- y shape is: (2, 10) 

- First observation is belong to class 1. It is a cat. 
- Second observation is belong to class 0. It is dog.
- Third observation is belong to class 0. It is a dog.

...

- Final observation is belong to class 1. It is a cat. 

We get our inputs: X, multiply them by their weights:W and each of them is summed up with bias:b.

- h represents the linear output.


    h = wx +b 

- we can predict the probabilities by using sigmoid function:

y_pred = sigmoid(h) --> gives us value btw 0-1. 

-Let's calculate probabilities for different classes.

    y_pred = [0.8  0.0  0.75.....0.6]

We find the probabilities of being in positive class, i.e class 1, or in our case it is being cat.


- First observation: class 1. It is predicted as a cat.  --> correct!
- Second observation: class 0. It is predicted as a dog. --> correct!
- Third observation: class 1. It is predicted as a cat.  ---> WRONG!

- Final observation: class 0. It is predicted as a dog. --> WRONG!


How can we calculate the errors?

    Error = - y * ln(y_pred) - (1-y) * ln(1-y_pred)


If y = 1 then we will use the first part of this error, if y=0 then we will use the second part of this error. Since y_pred
is calculated as the probability of being in class 1, the equation gives us the right probability for the case y_pred=0.

But how did we get this error function in the first place?

    Our aim is maximizing the product of probability: max(y_pred_1 * y_pred_2 * ... y_pred_10) 
    
    Note: if y = 0  use 1- y_pred 
          else      use y_prd


We need to turn it to sum. Product makes our life hard. 

    ln(y_pred_1 * y_pred_2 * ... y_pred_10) = ln(y_pred_1) + ... + ln(y_pred_10)
    
    Include the if/else part. And we get the cross entropy:
    
    Error = Sum(- y * ln(y_pred) - (1-y) * ln(1-y_pred))  ---> minimize this.
    
    
Now we need to update the weight & bias to get smaller cross entropy. 

    d --> derivative

    d/dy_pred  E * d/dh y_pred * d/dw h =  -(y-y_pred)*x

    d/dy_pred  E * d/dh y_pred * d/db h =  -(y-y_pred)
    

Then use these gradients to update the weight & bias:

    weight = weight - learning_rate * [ -(y-y_pred) * x]
    bias = bias - learning_rate * [-(y-y_pred)]
    
    
Quick example:

For 1st observation:

- y = 1
- w1 = 2  w2 = 1  b= 0 and x1 = 1  x2 = 0.225  
- h = w * x + b = 2.225
- sigmoid(h) = 0.8 ---> y_pred = 0.8

Calculate the gradient for w1 : -(y - y_pred)* x1 = -(1-0.8) * 1 = -0.2

Update the weights with lr = 1

w1 = w1 - 1*(-0.2) = 2.2 


Always go to the direction of reverse of gradient. If the probability of being in class 1 was smaller then the gradient would be higher.
And we need to increase the weights more. 