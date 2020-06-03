import numpy as np

wb1 = -3.37
wb2 = 0.125
w11 = -4
w12 = 1.5
w21 = -1.5
w22 = 0

n1 = 0.0
n2 = 0.0


def transferF(summ):
    return (2 / (1 + np.exp(-2 * summ))) - 1


# formular is like this : (OutputNeuron * wSelf) + (OutPutOtherNeuron * wOtherNeuron) + (1 * wBias)
def calcSumOfNeuron1(currentVal, n2):
    return currentVal * w11 + n2 * w21 + wb1 * 1


def calcSumOfNeuron2(currentVal, n1):
    return currentVal * w22 + n1 * w12 + wb2 * 1


d = input("start simultaenously? (y,n)")
print("d= ", d)

if 'y' in d:
    n1 = transferF(calcSumOfNeuron1(n1, 0))
    n2 = transferF(calcSumOfNeuron2(n2, 0))
    print("first values: n1 = ", n1, ", n2 = ", n2)

i = 1
while i < 20:
    n1 = transferF(calcSumOfNeuron1(n1, n2))
    n2 = transferF(calcSumOfNeuron2(n2, n1))
    print('t: ', i, ' Neuron1: ', n1, ' Neuron2: ', n2)
    i = i + 1
