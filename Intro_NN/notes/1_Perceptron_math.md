Example:

The function which separates the positive and negative classes is:

y = 3x1 + 4x2 -10

    However,point (4,5) is misclassified with this equation. It should be in negative area.  
    y = 3*4 + 4*5 -10 
    y = 22   
    y>= 0  ---> WRONG

Let's move the equation close to this point: 

| w1 | w2 | bias |
|---|----| --- |
| 3 | 4 | -10
| 4 | 5 | 1
| -1 | -1 | -11

now our equation becomes  y = (-x1) + (-x2) -11

this is a huge step. Instead we can use learning rate;

learning rate = 0.01

| w1 | w2 | bias |
|---|----| --- |
| 3 | 4 | -10
| 0.04 | 0.05 | 0.01
| 2.96 | 3.95 | -10.01

now our equation is y = 2.96 * x1 + 3.95 * x2 -10.01

[Perceptrons](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/intro_codes/Perceptrons.py)


## Perceptron Algorithm:

1. Start with random weights: w1, ..., wn, b

2. For every misclassified point (x1, ..., xn):

    2.1 if prediction = 0 (actual is 1):
    
     - wi = wi + &alpha; * xi
     - b = b + &alpha;
     for all i = 1...n where &alpha; is learning rate. 
     
    2.2. if prediction = 1 (actual is 0):
    
     - wi = wi - &alpha; * xi
     - b = b - &alpha;
     for all i = 1...n where &alpha; is learning rate. 
     
## Errror Function = Distance

Gradient Descent!

The error function can not be discrete, it should be continuous. If this was discrete, then we couldn't understand the direction.
Error function needs to be differentiable. 


Predictions are the answer we get from the algorithm. 

Discrete prediction is like; Yes! or No!
Continuous predictions are numbers. for example btw 0-1

For classification; there are probabilities that the points belong to a class. 
And there are distances from the lines.  Probability is a function of the distance from the line. 


Discrete prediction:
Step Function

    y = 1 if x>= 0 
        0 if x <0
        
Continuous prediction:
Sigmoid Function

    y (sigmoid(x)) = 1 / 1 + e^-x


example:

probabilities = [0.8, 0.7, 0.4, 0.1]

For the first point, probability of being blue (positive class or class 1) is 0.8 and probability of being red (negative class or class 0) is 0.2.

For the last point, probability of being blue is 0.1 and probability of being red (negative class or class 0) is 0.9.

### How can we get these probabilities?

First, we calculate the prediction from:

    y_pred = Wx + b

Then;

    y_probability = sigmoid(y_pred)
    
    i.e.
    
    y_probability = sigmoid(Wx + b)
    
    y_probability = 1 / 1 + e^ -(Wx + b)
   
Example:

The sigmoid function is defined as sigmoid(x) = 1/(1+e-x). 

If the score is defined by 4x1 + 5x2 - 9 = score, then which of the following points has exactly a 50% probability 
of being blue or red? 

Solution: 

We need to find the points which make the equation 0.

since; if x = 0, then 1 / 1 + e^ 0 = 0.5

(1,1) , (-4,5) are correct numbers.

[Perceptron_Algorithm](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/intro_codes/perceptron_algorithm.py)

## Why perceptron is not enough to solve XOR?

XOR function explanation: http://toritris.weebly.com/perceptron-5-xor-how--why-neurons-work-together.html

## Multiclass Classification & Softmax

Softmax --> the probabilities need to be 1 in sum.

Example:

- y1 = 2
- y2 = 1
- y3= 0

turning them to probabilities: 

- 2 / 2+1+0 = 2/3
- 1 / 2+1+0 = 1/3
- 0 / 2+1+0 = 0

summary is 1. 

There is a problem in this solution. what if our scores are negative? 

Example:

- y1 = -1
- y2 = 1
- y3= 0

turning them to probabilities: 

- 2 / -1+1+0 = NAN
- 1 / -1+1+0 = NAN
- 0 / -1+1+0 = NAN

 = (
 
 ⭐️ The exponential function only returns positive values !!
 
 ⭐️ Exponential function is the solution. 

Example:

- y1 = 2
- y2 = 1
- y3= 0

turning them to probabilities with exponential function:

- e^2 / e^2 + e^1 + e^0 = 0.67
- e^1 / e^2 + e^1 + e^0 = 0.24
- e^0 / e^2 + e^1 + e^0 = 0.09

summary is 1. This is softmax function. 

    Linear function scores = Z1, ..., Zn
    P(Class i ) = e^Zi / e^Z1 + ... + e^Zn

if the n = 2, then softmax ---> sigmoid. 

[Softmax_code](https://github.com/pelinbalci/Intro_Deep_Learning/blob/master/Intro_NN/intro_codes/softmax.py)
