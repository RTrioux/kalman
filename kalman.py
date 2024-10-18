import numpy as np
from numpy.typing import NDArray


class KF:

    def __init__(self, A: NDArray, B: NDArray, C: NDArray, D: NDArray, R: NDArray, Q: NDArray, mu0=None, sigma0=None):

        # State Space model
        self.A = np.copy(A)
        self.B = np.copy(B)
        self.C = np.copy(C)
        self.D = np.copy(D)

        # Covariance matrices
        self.R = np.copy(R)  # Process Covariance Xdot = AX + Bu + epsilon (Epsilon's covariance)
        self.Q = np.copy(Q)  # Measurement Covariance Y = CX + delta (delta's covariance)
        self.Q_init = np.copy(Q)  # Save Initial covariance matrix

        # States
        p, n = C.shape
        self.mu = np.zeros((n, 1))  # Mean i.e the center of the prediction
        self.sigma = np.zeros((n, n))  # Covariance matrix
        if mu0 is not None:
            self.mu = mu0
        if sigma0 is not None:
            self.sigma = sigma0

        self.mu_pred = np.zeros((n, 1))
        self.sigma_pred = np.zeros((n, n))

        self.z = np.zeros((p, 1))  # We need to store measurement to hold to previous measurement in case of data loss

        # DEBUG
        self.K = np.zeros((n, p))

    def step(self, u: NDArray, z: NDArray, w: list = None):

        p, n = self.C.shape
        # W = np.eye(p, p)
        if w is not None:
            for idx, wi in enumerate(w):
                if wi is False:
                    self.Q[idx, idx] = 1e6
                else:
                    self.Q[idx, idx] = self.Q_init[idx, idx]

        # mouse_buttons = pygame.mouse.get_pressed()
        # if mouse_buttons[0]:
        # self.Q = 1e6 * np.array([[1, 0], [0, 1]])
        # else:
        # self.Q = 4 * np.array([[1, 0], [0, 1]])
        # if mouse_buttons[2]:
        # pass

        self.predict(u)
        self.correct(z, w)

    def predict(self, u: NDArray):
        A = self.A
        B = self.B
        R = self.R
        mu = self.mu
        sigma = self.sigma

        self.mu_pred = A @ mu + B @ u
        self.sigma_pred = A @ sigma @ A.T + R

    def correct(self, z: NDArray, w: list = None):
        z = np.copy(z)  # TODO: Check if that's requiered
        if w is not None:
            for idx, wi in enumerate(w):
                if wi is True:  # If we have a measurement available, update it
                    self.z[idx, 0] = z[idx, 0]
        z = self.z  # For more convenient use in equations

        Q = self.Q
        C = self.C
        n = self.A.shape[0]
        sigma_pred = self.sigma_pred
        mu_pred = self.mu_pred

        K = sigma_pred @ C.T @ np.linalg.inv(C @ sigma_pred @ C.T + Q)  # Solution of the Algebric Ricati Equation
        self.K = K  # DEBUG
        self.mu = mu_pred + K @ (z - C @ mu_pred)
        self.sigma = (np.eye(n, n) - K @ C) @ sigma_pred
