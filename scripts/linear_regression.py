"""Linear regression model from scratch using batch gradient descent."""

import numpy as np
from utils import generate_random_linear_data

class LinearRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 1000):
        """
        Init the Linear regression model. 
        Params: 
            - lr (float): Learning rate for GD algorithm (default = 0.1)
            - epochs (int): Total of iterations for the training (default = 1000)
        """
        self.lr = lr
        self.epochs = epochs
        self.theta = None
        self.bias = None

    def fit(self): 
        pass

    def predict(self): 
        pass

    def evalueate(self): 
        pass

if __name__ == "__main__":
    X, y, theta_true = generate_random_linear_data()
    LR = LinearRegression()
    pass