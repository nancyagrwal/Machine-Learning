import numpy as np
from calculateZ import WeightCrossData, AddBiasToProduct
from output import Softmax_activation
from activationFunctions import Tanh,Relu,Sigmoid

class FeedForwardNeuralNetwork:
    def __init__(self,S):
        self.biases = []
        self.Weights = []
        for i in range(len(S)-1):
            self.Weights.append(np.random.randn(S[i], S[i+1]) / np.sqrt(S[i]))
            self.biases.append(np.random.randn(S[i+1]).reshape(1, S[i+1]))

    def findLoss(self, X, y,activFun):
        dotProduct = WeightCrossData()
        adddBias = AddBiasToProduct()
        if(activFun == "tanh"):
            activationFunctions = Tanh()
        elif(activFun == "sigmoid"):
            activationFunctions = Sigmoid()
        else:
            activationFunctions = Relu()
     
        softmax_activation = Softmax_activation()

        inputTaken = X
        for i in range(len(self.Weights)):
            dp = dotProduct.prop_forward(self.Weights[i], inputTaken)
            zobt = adddBias.prop_forward(dp, self.biases[i])
            inputTaken = activationFunctions.prop_forward(zobt)

        return softmax_activation.calculate_loss(inputTaken, y)

    def predict(self, X,activFun):
        dotProduct = WeightCrossData()
        adddBias = AddBiasToProduct()
        if(activFun == "tanh"):
            activationFunctions = Tanh()
        elif(activFun == "sigmoid"):
            activationFunctions = Sigmoid()
        else:
            activationFunctions = Relu()
          
        softmax_activation = Softmax_activation()
        inputTaken = X
        for i in range(len(self.Weights)):
            dp = dotProduct.prop_forward(self.Weights[i], inputTaken)
            zobt = adddBias.prop_forward(dp, self.biases[i])
            inputTaken = activationFunctions.prop_forward(zobt)
    
        return np.argmax(softmax_activation.predict(inputTaken), axis=1)

    def train(self, X, y, activFun , max_passes=20000, epsilon=0.01, reg_lambda=0.01, loss_printYN=False):
        dotProduct = WeightCrossData(); adddBias = AddBiasToProduct()
        if(activFun == "tanh"):
            activationFunctions = Tanh()
        elif(activFun == "sigmoid"):
            activationFunctions = Sigmoid()
        else:
            activationFunctions = Relu()
        softmax_activation = Softmax_activation()

        for iter in range(max_passes):
            # perform forward propagation
            inputTaken = X
            prop_forward = [(None, None, inputTaken)]
            for i in range(len(self.Weights)):
                dp = dotProduct.prop_forward(self.Weights[i], inputTaken)
                zobt = adddBias.prop_forward(dp, self.biases[i])
                inputTaken = activationFunctions.prop_forward(zobt)
                prop_forward.append((dp, zobt, inputTaken))

            # Back propagation
            dactivFunc = softmax_activation.calculate_diff(prop_forward[len(prop_forward)-1][2], y)
            
            for n in range(len(prop_forward)-1, 0, -1):
                add_descent = activationFunctions.prop_backward(prop_forward[n][1], dactivFunc)
                bias_descent, mul_descent = adddBias.prop_backward(prop_forward[n][0], self.biases[n-1], add_descent)
                weight_descnt, dactivFunc = dotProduct.prop_backward(self.Weights[n-1], prop_forward[n-1][2], mul_descent)
                weight_descnt += reg_lambda * self.Weights[n-1]
                self.biases[n-1] += -epsilon * bias_descent
                self.Weights[n-1] += -epsilon * weight_descnt

            if loss_printYN and iter % 1000 == 0:
                print("Loss after iteration %i: %f" %(iter, self.findLoss(X, y,activFun)))
        return self.Weights,self.biases;        