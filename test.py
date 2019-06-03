import numpy as np
import CNN



weights = []

weight = {
    'w' : np.ones((5,5)),
    'b' : np.zeros((5,1))

}

weights.append(weight)


layer_weights = {
    'conv' : weights,
    'fullconnect': None

}



t = np.arange(9).reshape((3,3))
tt = np.arange(9).reshape((3,3))

ttt = np.zeros((2,3,3))


In = np.arange(1,4).reshape((1,3,1))
W = np.ones((2,3))
B = np.ones((2,1))
W[:,1] += 1
W[:,2] += 2

WE = {
    'W' : W,
    'B' : B
}

cnn = CNN.ConvNN()

print(cnn.sigmoid(8.04)*2)


res = cnn.forwardPass(np.ones((1,16,16,1)))
print(res[0])









