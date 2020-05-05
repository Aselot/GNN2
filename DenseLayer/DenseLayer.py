import numpy as np


class DenseLayer:
    """
    Biases init normalerweise 0, aber falls ein Neuron nicht aktiviert wird, dann sollte Bias > 0
    Weights moeglichst kleiner Range ist besser, aber hoechstens von [1,-1]
    """

    def __init__(self, n_inputs, n_neurons):
        """Initialisiert das Dense Layer und die dazu benoetigten Weights und Biases

        Arguments:
            n_inputs {int} -- Eingehende Matrix, koennten die Eingabewerte sein, oder die Ausgabewerte eines anderen Neurons
            n_neurons {int} -- Anzahl der Neuronen, die von der Eingabe lernen sollen

        >>> # MNIST Sample
        >>> layer1 = DenseLayer(784, 400)
        >>> output1 = layer1.forward(MNIST_DATA)
        >>> layer2 = DenseLayer(output1, 10)
        >>> output2 = layer2.forward(output1)
        """
        # Shape mit (n_inputs, n_neurons)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # Vektor mit shape (Einer Zeile)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
