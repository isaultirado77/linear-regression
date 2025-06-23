import numpy as np


def generate_random_linear_data(theta_0: float = 4,
                                theta_1: int = 3,
                                n_samples: int = 100,
                                noise_std: float = 1.0,
                                seed: int = 0):
    """
    Genera datos sintéticos para regresión lineal: y = theta_0 + theta_1 * x + ruido

    Retorna:
    - X_b: matriz de características con columna de bias incluida (matriz de diseño)
    - y: vector objetivo
    - true_theta: vector con los parámetros reales [theta_0, theta_1]
    """

    rng = np.random.default_rng(42)   # Generador aleatorio con semilla de reproducibilidad fija

    sigma = 1.0  # stdev del ruido
    X = 2 * rng.random(size=(n_samples, 1))  # valores de x entre 0 y 2
    true_theta = np.array([[theta_0], [theta_1]])  # intercepto, pendiente
    noise = rng.normal(0, noise_std, size=(n_samples, 1))
    y = true_theta[0] + true_theta[1] * X + noise
    X_b = np.c_[np.ones(n_samples).reshape(-1, 1), X]
    return X_b, y, true_theta
