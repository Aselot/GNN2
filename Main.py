import numpy as np
from DenseLayer import DenseLayer
from Utils.DataSetGenerator import create_data
from ActivationFuncs.ActivationFuncs import ReLu, SoftMax
from Loss.CategoricalCrossEntropy import CategoricalCrossEntropy

np.random.seed(0)

if __name__ == '__main__':
    X, y = create_data(100, 3)
    activation1 = ReLu()
    dense1 = DenseLayer(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs
    # Make a forward pass of our training data through this layer
    dense1.forward(X)
    # Use activation on output of the first dense layer
    activation1.forward(dense1.output)
    dense2 = DenseLayer(3, 3)
    dense2.forward(dense1.output)
    # Softmax fuer Klassizifierung
    activation2 = SoftMax()
    activation2.forward(dense2.output)

    # Define loss
    loss_function = CategoricalCrossEntropy()
    loss = loss_function.forward(activation2.output, y)
    print 'loss: ', loss

    # Calculate accuracy from output of activation2 and targets
    predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis (column=
    accuracy = np.mean(predictions == y)


    print 'accuracy: ', accuracy
