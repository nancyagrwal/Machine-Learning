import numpy as np

class Softmax_activation:
    
    def predict(self, X):
        scr = np.exp(X)
        return scr / np.sum(scr, axis=1, keepdims=True)

    def calculate_loss(self, X, y):
        data_count = X.shape[0]
        ypred = self.predict(X)
        corect_logypred = -np.log(ypred[range(data_count), y])
        loss_obtained = np.sum(corect_logypred)
        return 1./data_count * loss_obtained

    def calculate_diff(self, X, y):
        data_count = X.shape[0]
        ypred = self.predict(X)
        ypred[range(data_count), y] -= 1
        return ypred