import random

import pygame as pyg
import numpy as np

bias = 1
lernrate = 0.2
numPatterns = 300

w_Input_Hidden_Factor = np.random.uniform(-0.7, 0.7, 12)
w_Hidden_Output_Factor = np.random.uniform(-0.7, 0.7, 5)




# bedenke, target ist die länge des vektors: squared(x²+y²)<=1 => 0.8, else 0


def getNeuralHiddenOutput(x, y):
    arr = np.empty(4)
    j = 0
    for i in range(0, 4):
        arr[i] = 1.0 / (1.0 + np.exp(-5 * (bias * w_Input_Hidden_Factor[j] + x * w_Input_Hidden_Factor[j + 1] + y * w_Input_Hidden_Factor[j + 2])))
        j += 3
    return arr


def getNeuralOutput(o2, o3, o4, o5):
    summ = bias * w_Hidden_Output_Factor[0] + w_Hidden_Output_Factor[1] * o2 + w_Hidden_Output_Factor[2] * o3 + w_Hidden_Output_Factor[3] * o4 + w_Hidden_Output_Factor[4] * o5

    return 1.0 / (1.0 + np.exp(-5 * summ))


def train():
    # generate #numPatterns test cases:
    xxt = np.random.uniform(-1.2, 1.2, numPatterns)
    yyt = np.random.uniform(-1.2, 1.2, numPatterns)

    # calculate length of the test cases and put solution in 0.8 or 0
    sst = np.empty(numPatterns)
    target = np.empty(numPatterns)

    for x in range(0, numPatterns):
        t = np.sqrt(xxt[x] * xxt[x] + yyt[x] * yyt[x])
        sst[x] = t
        if t >= 1:
            target[x] = 0
        else:
            target[x] = 0.8

    #train for each trainingsunit (pattern)
    for p in range(numPatterns):

        output_hidden = getNeuralHiddenOutput(xxt[p], yyt[p])
        output_end = getNeuralOutput(output_hidden[0], output_hidden[1], output_hidden[2], output_hidden[3])

        global w_Hidden_Output_Factor
        global w_Input_Hidden_Factor

        w_Hidden_Output_Factor[0] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * bias
        w_Hidden_Output_Factor[1] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * output_hidden[0]
        w_Hidden_Output_Factor[2] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * output_hidden[1]
        w_Hidden_Output_Factor[3] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * output_hidden[2]
        w_Hidden_Output_Factor[4] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * output_hidden[3]

        # wHiddenfactor new target,

        i = 0

        for x in range(0, 4):
            w_Input_Hidden_Factor[i] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * w_Hidden_Output_Factor[x + 1] * output_hidden[x] * (1.0 - output_hidden[x]) * bias
            w_Input_Hidden_Factor[i + 1] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * w_Hidden_Output_Factor[x + 1] * output_hidden[x] * (1.0 - output_hidden[x]) * xxt[p]
            w_Input_Hidden_Factor[i + 2] += lernrate * (target[p] - output_end) * output_end * (1.0 - output_end) * w_Hidden_Output_Factor[x + 1] * output_hidden[x] * (1.0 - output_hidden[x]) * yyt[p]
            i += 3


pyg.init()
pyg.display.set_caption('Rechner')
screen = pyg.display.set_mode((478, 478))
gen=0
while True:
    gen+=1
    train()
    for x in np.arange(-1.2, 1.2, 0.005):
        for y in np.arange(-1.2, 1.2, 0.005):
            pyg.event.get()
            vec = np.sqrt(x * x + y * y)
            outputHdden = getNeuralHiddenOutput(x, y)
            output = getNeuralOutput(outputHdden[0], outputHdden[1], outputHdden[2], outputHdden[3])
            color = (int(output * 255), int(output * 255), int(output * 255))
            coord = int((x+1.2)*200  ), int((y+1.2)*200  )
            pyg.draw.rect(screen, color, (coord[0], coord[1], 2, 2))
        pyg.display.flip()
        print("vector len = ", vec, ", output= ", output, ",      ", "color: ", color, ", cord: ", coord,", gen: ",gen)
