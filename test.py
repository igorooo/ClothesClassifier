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


np.set_printoptions(suppress=True)

res,val = cnn.forwardPass(np.ones((1,16,16,1)))

label = np.zeros((2,1))
label[0,0] = 1

G = cnn.backwardPass(label, res, val)

#print(G['conv'][1])

test = np.arange(256).reshape((1,16,16,1))

print(cnn.sigmoid(cnn.sigmoid(5)*4 +1)*2 +1)






















