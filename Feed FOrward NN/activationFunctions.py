import numpy as np

# hyperbolic tangent activation function
class Tanh:
    def prop_forward(self, X):
        return np.tanh(X)

    def prop_backward(self, X, grd_param):
        output = self.prop_forward(X)
        return (1.0 - np.square(output)) * grd_param
    
# sigmoid activation function  
class Sigmoid:
    def prop_forward(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def prop_backward(self, X, grd_param):
        output = self.prop_forward(X)
        return (1.0 - output) * output * grd_param

# linear activation function
class Relu:
    def prop_forward(self,X):
        return np.log(1+ np.exp(X))
    
    def prop_backward(self,X,grd_param):
        return (1.0 / (1.0 + np.exp(-X))) * grd_param
             
