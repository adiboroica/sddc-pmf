import pickle


class Data:
    def __init__(self, k1, k2, Fx, Fy, probability_matrix, rho_x, rho_y):
        self.k1 = k1
        self.k2 = k2
        self.Fx = Fx
        self.Fy = Fy
        self.probability_matrix = probability_matrix
        self.rho_x = rho_x
        self.rho_y = rho_y

    def save(self, path, verbose=False):
        with open(path, "wb") as f:
            pickle.dump(self, f)

        verbose and print(f"Data saved to {path}!")

    @staticmethod
    def load(path, verbose=False):
        with open(path, "rb") as f:
            data = pickle.load(f)

        verbose and print(f"Data loaded from {path}!")

        return data
