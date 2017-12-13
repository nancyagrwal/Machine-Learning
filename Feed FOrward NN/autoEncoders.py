import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy.io as scio
    
mat = scio.loadmat('./ExtYaleB10.mat')
Y_test = mat['test']
Y_train = mat['train']

def imageResizing(data):
    resizedDatasET = []
    for img in data:
        resizedDatasET.append(resize(img, (20, 17), mode='constant'))
    resizedDatasET = np.array(resizedDatasET)
    return resizedDatasET

def imageReshaping(data):
    dimension = data.shape[1] * data.shape[2]
    return data.reshape(data.shape[0], dimension)

def findError(y, ypred):
    return (ypred != y).sum().astype(float) / len(ypred)

def inputProcessing(data):
    X = [];Y = []
    for i in range(len(data[0])):
        people_count = data[0][i].T
        for j in range(len(people_count)):
            X.append(people_count[j].T);Y.append(i)
    X = np.array(X);Y = np.array(Y)
    fig, axis = plt.subplots(figsize=(12,8))
    axis.imshow(X[1], cmap='gray')
    X = imageResizing(X)
    X = imageReshaping(X)
    return X, Y

X,Y = inputProcessing(Y_train)
Xtst,Ytst = inputProcessing(Y_test)
maximums = np.amax(X, axis=0)
minimums = np.amin(X, axis=0) 
ActivationData = (X - minimums) / (maximums - minimums)
ActivationData = ActivationData.transpose() 
  
     
def new_layer(dimension_in, dimension_out):
        weights = np.random.rand(dimension_out, dimension_in) - 0.5
        bias = np.random.rand(dimension_out, 1) - 0.5
        
        return {
            'weights': weights,
            'bias': bias,
            'act': np.zeros((dimension_out, ActivationData.shape[1])),
            'grad_activations': np.zeros((dimension_out, ActivationData.shape[1])),
            'act_errors': np.zeros((dimension_out, ActivationData.shape[1])),
            'grad_weights': np.zeros(weights.shape),
            'grad_biases': np.zeros(bias.shape)
        }

def createinputLayers():
        s1 = 340; s2 = 100; s3=10;
        return [new_layer(s1, s2), new_layer(s2, s3), new_layer(s3, s2), new_layer(s2, s1)]

    
def forward_propagation(inputLayers,inputData,actFn):
        for layer in inputLayers:
            Z = np.matmul(layer['weights'], inputData) + layer['bias']
            act, grad_activations = actFn(Z)
            layer['act'] = act
            layer['grad_activations'] = grad_activations
            inputData = act

        
def backward_propagation(inputLayers, inputData, outputData):    
      
        act_errors = (inputLayers[-1]['act'] - outputData)
        for index_layer, layer in reversed(list(enumerate(inputLayers))):
            layer['act_errors'] = act_errors
            hadamard = act_errors * layer['grad_activations']
            if index_layer == 0:
                last_activations = inputData
            else:
                last_activations = inputLayers[index_layer - 1]['act']
            grad_weights = np.matmul(hadamard, last_activations.transpose()) / inputData.shape[1]
            layer['grad_weights'] = grad_weights
            grad_biases = hadamard.sum(axis=1).reshape(layer['bias'].shape) / inputData.shape[1]
            layer['grad_biases'] = grad_biases
            if index_layer != 0:
                act_errors = np.matmul(layer['weights'].transpose(), hadamard)
            
def gd(inputLayers, top_diff):
        for layer in inputLayers:
            layer['weights'] = layer['weights'] - layer['grad_weights'] * top_diff
            layer['bias'] = layer['bias'] - layer['grad_biases'] * top_diff 
            
            
def computeError(inputLayers):
        act_errors = inputLayers[-1]['act_errors']
        e_squared = act_errors ** 2
        return e_squared.sum() / act_errors.shape[1]  

def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    return sigmoid, sigmoid_grad


def relu(x):
    relu = np.maximum(x, np.zeros(x.shape))
    relu_grad = np.ones(x.shape) * (x > 0)
    return relu, relu_grad

def tanh( X):
     return np.tanh(X), (1.0 - np.square(np.tanh(X))) 
    
def trainAE(actFn, epsilon, iter):
        inputLayers = createinputLayers()
        arrayErr = []
        for epoch in range(iter):
            forward_propagation(inputLayers, ActivationData, actFn)
            backward_propagation(inputLayers, ActivationData, ActivationData)
            gd(inputLayers, epsilon)
            arrayErr.append(computeError(inputLayers))
        return inputLayers, arrayErr

for actfn in [relu,sigmoid,tanh]:
    inputLayers, arrayErr = trainAE(actfn, 0.01, 1000)
    print("Weight, bias and activations are..." , inputLayers)
    plt.plot(range(1000), arrayErr)
    plt.show()
    ec1 = np.array(inputLayers[1]['act'][0, :].transpose())
    ec2 = np.array(inputLayers[1]['act'][1, :].transpose())
    plt.scatter(ec1, ec2 , c = Y)
    plt.show()
    print(X.shape)

   
   