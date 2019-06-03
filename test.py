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

#print(cnn.sigmoid(cnn.sigmoid(5)*8+1)*4+1)


res = cnn.forwardPass(np.ones((1,16,16,1)))
label = np.zeros((2,1))
label[0,0] = 1


res2 = cnn.backwardPass(label, res[0], res[1])

#print(res[0])

W = np.ones((2,4))
dZ = np.array([-0.5,0.5]).reshape((2,1))

#print(W.T@dZ)

Wconv = np.zeros((3, 3, 1, 2))
Wconv[1, 1, :, :] = 1

wConv1 = {
    'W': Wconv,
    'B': np.ones((2, 1)),
}


In = np.ones((1,16,16,1))
dZ = np.ones((1,14,14,2))

bRes = cnn.convolution_layer_backward(dZ,wConv1,In)
np.set_printoptions(suppress=True)
print(bRes[0].shape)
print(bRes[0][0,:,:,0])












