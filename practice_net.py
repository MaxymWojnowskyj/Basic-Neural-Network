from venv import create
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


def createLayers(input_count, neuron_counts):
    # cL(2, [4, 4, 3]) will create 2*4 weights with 4 bias's for first layer then 4*4 + 4 then 4*3 + 3
    weights, biases = [], []
    #input_count = input_count
    for neuron_count in neuron_counts:

        weights.append(0.01 * np.random.randn(input_count, neuron_count))
        biases.append(np.zeros((1, neuron_count)))
        input_count = neuron_count

    return weights, biases


def layerForwardPass(inputs, weights, biases, layer):
    # index out weights only from layer index specified
    return np.dot(inputs[layer], weights[layer]) + biases[layer]

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # subtract each rows max to prevent exploding values (all values will be between: e^-inf to e^0 )
    exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    #normalize values
    norm = exp/np.sum(exp, axis=1, keepdims=True)

    return norm

def logCrossEntLoss(outputs, y):
    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clip = np.clip(outputs, 1e-7, 1 - 1e-7)
    # we are doing one hot encoding so we can simplify -sum: true*log(pred) to -1*log(pred) (since the other values will be zero)
    # range(outputs) gives a list of indexing rows (we take range as we want to index each row) we then take the index value stored in y of each row
    # ex: y_pred = [ [0.2, 0.3, 0.5], [0.1, 0.7, 0.2]] y = [2, 1]  range(len(outputs)) = [0, 1] y_pred[range(len(outputs)), y] = [0.5, 0.7]
    correct_pred = y_pred_clip[range(len(outputs)), y]
    # take the negative log of each of the confidence levels to get error for each prediction
    return np.mean(-np.log(correct_pred))


def derivForward(inputs, weights, dvalues):
    # inputs = matrix each row is one sample of inputs. size: (number_of_samples x input_nuerons)
    # dvalues = matrix each row is the derivative of each nueron in a layer. size (number_of_samples x nuerons_in_forward_layer)
    # Ex: for input nueron 1 multiply its weight with derivative from that sample then sum all these multiplications across the number of samples (w0*dz0 + w0*dz0 + w0*dz0 + ...) (each is one instance sample from a batch of inputs)
    # since deriv w/ resp to bias is just a const the deriv is 1 so we are multiply dvalues with 1 then summing so can just sum and skip multiply by 1
    # axis=0 means summing the columns as each column is multiply sample instances of the same bias (the row contains one instance of biases on a layer)
    dinputs = np.dot(dvalues, weights.T)
    dweights = np.dot(inputs.T, dvalues)
    dbiases = np.sum(dvalues, axis=0, keepdims=True) 
    return dinputs, dweights, dbiases


def derivReLU(Z, dvalues):
    # relu = max(0, z)
    # deriv relu = 1(z>0) two cases: d/dz (z <= 0) then relu(z) = 0 thus d/dz(0) = 0, other case (z > 0) then relu=z thus d/dz(z) = 1
    # so we will be left with drelu = matrix of 1's and 0's we are multiplying by deriv of forward layer deriv values (dvalues) since dvalue*1 = dvalue and dvalue*0 = 0 we can just modify all dvalues to 0 where their z value is less than 0
    drelu = dvalues.copy()
    drelu[Z <= 0] = 0
    return drelu


def derivLogCEL(outputs, y):
    # i current sample, j = index of prediction (predicing from range of classes)
    # d/dpredy(-sumj yi,j*ln(pred_yi,j)) = -sumj yi,j d/dpred_y(ln(pred_yi,j)) = -sum yi,j 1/pred_yi,j d/dpred_y(pred_yi,j) = -sum yi,j 1/pred_yi,j * 1 = yi,j/pred_yi,j 
    # actual y value (in this case 1 or 0) divided by the prediction value (a fraction/percentage)
    samples = len(outputs) # number of samples
    labels = len(outputs[0]) # range of classes to predict from
    # lets say labels = 3 this takes y = [0, 1, 0, 2] and creates y_vector = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
    y_vector = np.eye(labels)[y]
    # this is done so we can divide row wise the true values by the pred values (calculating the gradient)
    dinputs = -y_vector / outputs
    # the more samples we have the larger our gradient will become as we are summing them across samples so to prevent large and varying values we normalize it
    dinputs = dinputs / samples

    return dinputs

def derivLogCELnSoftmax(outputs, y):
    # optimized combines deriv of log cross entropy loss and softmax to make a simpler and faster deriv
    samples = len(outputs)
    dinputs = outputs.copy() # create a copy so we dont overwrite outputs
    # subtract one from each samples correct y prediction confidence value 
    dinputs[range(samples), y] -= 1 # gradient of log cross entr w/ respect to softmax: pred_y - true_y
    dinputs = dinputs / samples # normalize the gradient
    return dinputs

def accuracy(outputs, y):
    predictions = np.argmax(outputs, axis=1) #take the highest prediction from each row
    accuracy = np.mean(predictions==y)
    return accuracy

def fullPass(X, W, B):
    inputs = [X] # first 5 samples make array as we will be saving inputs into hidden layers that already have relu and mods from previous layers applied

    #forward pass and ReLU activate for all layers up to the second last layer
    for layer_idx in range(len(W)-1):
        Z = layerForwardPass(inputs, W, B, layer_idx) 
        inputs.append(ReLU(Z))

    #forward pass to the output layer
    Z = layerForwardPass(inputs, W, B, layer_idx+1)
    # apply output layer activation (softmax) 
    outputs = softmax(Z)

    return inputs, outputs

def backProp(inputs, outputs, W, y):
    dinputs = [derivLogCELnSoftmax(outputs, y)]
    dweights = []
    dbiases = []
    #dinputs.append(derivLogCELnSoftmax(sample_y, outputs))
    d_count = 0
    for layer_idx in reversed(range(len(W))):
        dinput, dweight, dbias = derivForward(inputs[layer_idx], W[layer_idx], dinputs[d_count])
        dweights.append(dweight)
        dbiases.append(dbias)
        if layer_idx != 0:
            dinputs.append(derivReLU(inputs[layer_idx], dinput))
            #print('dinputs at', layer_idx, get_dims(derivReLU(inputs[layer_idx], dinput)))
            d_count+=1
    
    return dweights, dbiases

def lr_decay(epochs, lr, decay):
    return lr * (1. / (1. + decay * epochs))

def momentumUpdate(w, b, vdw, vdb, dw, db, lr):
    mom_rate = 0.5
    new_vdw = mom_rate*vdw + (1-mom_rate)*dw
    new_vdb = mom_rate*vdb + (1-mom_rate)*db
    new_w = w - lr*new_vdw
    new_b = b - lr*new_vdb
    return new_w, new_b, new_vdw, new_vdb

def optimize(X, y, W, B):
    epochs = 10001
    lr = 1
    for epoch in range(epochs):
        inputs, outputs = fullPass(X, W, B)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
            f'acc: {accuracy(outputs, y):.3f}, ' +
            f'loss: {logCrossEntLoss(outputs, y):.3f}')
        dweights, dbiases = backProp(inputs, outputs, W, y) 

        #lr = lr_decay(epoch, lr, 1e-3)
        for w, dw, b, db in zip(W, dweights[::-1], B, dbiases[::-1]):
            w += -lr*dw
            b += -lr*db

        # reverse order of deriv array since we appended during backprop (going backwards) and weights and biases array was created by going forwards
        #dweights = dweights[::-1]
        #dbiases = dbiases[::-1] 
        # initialize vdw and vdb to contain a zero filled matrix for holding the previous derivative for the specific layer
        #vdw = [np.zeros((len(dwi), len(dwi[0]))) for dwi in dweights]
        #vdb = [np.zeros((len(dbi), len(dbi[0]))) for dbi in dbiases] 
        #for i in range(len(W)):
            #w = W[i]
            #b = B[i]
            #dw = dweights[i]
            #db = dbiases[i]
            #W[i], B[i], vdw[i], vdb[i] = momentumUpdate(w, b, vdw[i], vdb[i], dw, db, lr)
    
    return W, B


def main():
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    # X = [ [x-coord, y-coord], [x-coord, y-coord] ]
    # y = [ [class], [class] ] (class can be either 0, 1, or 2) (based on classes=3)
    W, B = createLayers(2, [64, 3])
    W, B = optimize(X, y, W, B)
    X_test, y_test = spiral_data(samples=25, classes=3)
    inputs, outputs = fullPass(X_test, W, B)
    print(f'test acc: {accuracy(outputs, y_test):.3f}, ' +
    f'test loss: {logCrossEntLoss(outputs, y_test):.3f}') 


main()