import numpy as np

class CategoricalCrossEntropy:
    def __init__(self):
        pass

    def forward(self, y_pred, y):
        n_samples = len(y_pred)

        # Probabilities for target values (y_pred=[0.30,0.2,0.5] y=[1,0,0]) => 0.3
        y_pred = y_pred[range(n_samples), y]

        # Loss fuer jedes einzelne Element
        negative_log_likelihoods = -np.log(y_pred)

        # Mean Loss
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss