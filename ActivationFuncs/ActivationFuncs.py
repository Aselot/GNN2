import numpy as np


class ReLu:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


class SoftMax:
    def __init__(self):
        self.output = None
        pass

    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # axis = 1 damit ueber die Zeilen hinweg
        # berechnet wird, bei batches also ueber jedes sample einzeln
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # s.o.
        # aufgrund der Gegebenheiten, dass beim Exponenten gegen -unendlich und unendlich die Werte zu 0 und 1.0
        # tendieren, kann beliebig subtrahiert und addiert werden ohne die Wahrscheinlichkeiten zu beeinflussen
        self.output = probabilities
        return self.output
