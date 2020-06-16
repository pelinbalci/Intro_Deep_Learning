import numpy as np


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function. e^x1 / e^(x1+x2+..+xn)
def softmax(L):
    proba_list = []
    proba_logsoftmax_list = []
    convert_to_prob_list = []
    total_exp = 0
    for i in range(len(L)):
        total_exp += np.exp(L[i])
    for i in range(len(L)):
        prob = np.exp(L[i]) / total_exp
        prob_logsoftmax = np.log(prob)
        convert_to_prob = np.exp(prob_logsoftmax)

        proba_list.append(prob)
        proba_logsoftmax_list.append(prob_logsoftmax)
        convert_to_prob_list.append(convert_to_prob)
    return proba_list, proba_logsoftmax_list, convert_to_prob_list


L = [2, 1, 0]
proba_list, proba_logsoftmax_list, convert_to_prob_list = softmax(L)
print('my_way:', proba_list)
print('logsoftmax:', proba_logsoftmax_list)
print('back to prob:', convert_to_prob_list)


def softmax_sol(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i * 1.0 / sumExpL)
    return result

    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())


L2 = [2, 1, 0]
result = softmax_sol(L2)
print('right_way', result)