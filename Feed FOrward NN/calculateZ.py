import numpy as np

class WeightCrossData:
    def prop_forward(self,Weights,data):
        return np.dot(data, Weights)

    def prop_backward(self, Weights,data, z_descnt):
        weight_descnt = np.dot(np.transpose(data), z_descnt)
        x_descnt = np.dot(z_descnt, np.transpose(Weights))
        return weight_descnt, x_descnt

class AddBiasToProduct:
    def prop_forward(self, data, b):
        return data + b

    def prop_backward(self, data, b, z_descnt):
        x_descnt = z_descnt * np.ones_like(data)
        b_descnt = np.dot(np.ones((1, z_descnt.shape[0]), dtype=np.float64), z_descnt)
        return b_descnt, x_descnt