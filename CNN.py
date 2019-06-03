import numpy as np



class ConvNN():

    gauss_MEAN = 0
    gauss_ST_DEVIATION = 1

    """
    WEIGHTS STRUCTURE EXAMPLE :


    weights = []

    weight = {
        'W': np.ones((5, 5)),
        'B': np.zeros((5, 1))

    }

    weights.append(weight)

    layer_weights = {
        'conv': weights,
        'fullconnect': None
        'softmax' : None

    }

    print(layer_weights['conv'][0]['B'])

    """



    def __init__(self):
        self.weights = None
        self.weights_grad = None
        #self.init_test_weights()
        self.init_random_weights()
        pass

    def init_test_weights(self):

        ConvLayersWeights = []
        FFLayersWeights = []

        Wconv = np.zeros((3,3,1,2))
        Wconv[1, 1, :, :] = 1



        wConv1 = {
            'W': Wconv,
            'B': np.ones((2,1)),
        }

        Wconv = np.zeros((3,3,2,2))
        Wconv[1, 1, :, :] = 1

        wConv2 = {
            'W': Wconv,
            'B': np.ones((2,1)),
        }

        wFF1 = {
            'W': np.ones((4,8)),
            'B': np.ones((4,1)),
        }

        wFF2 = {
            'W': np.ones((2,4)),
            'B': np.ones((2,1)),
        }

        ConvLayersWeights.append(wConv1)
        ConvLayersWeights.append(wConv2)
        FFLayersWeights.append(wFF1)
        FFLayersWeights.append(wFF2)

        layer_weights = {
            'conv': ConvLayersWeights,
            'fullyconnect': FFLayersWeights
        }

        self.weights = layer_weights


    def init_random_weights(self):

        ConvLayersWeights = []
        FFLayersWeights = []


        wConv1 = {
            'W': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (3,3,1,8)),
            'B': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (8,1)),
        }

        wConv2 = {
            'W': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (3,3,8,8)),
            'B': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (8,1)),
        }

        wConv3 = {
            'W': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (3,3,8,8)),
            'B': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (8,1)),
        }

        wConv4 = {
            'W': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (3,3,8,8)),
            'B': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (8,1)),
        }

        wFF1 = {
            'W': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (100,288)),
            'B': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (100,1)),
        }

        wFF2 = {
            'W': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (10,100)),
            'B': np.random.normal(ConvNN.gauss_MEAN, ConvNN.gauss_ST_DEVIATION, (10,1)),
        }

        ConvLayersWeights.append(wConv1)
        ConvLayersWeights.append(wConv2)
        ConvLayersWeights.append(wConv3)
        ConvLayersWeights.append(wConv4)
        FFLayersWeights.append(wFF1)
        FFLayersWeights.append(wFF2)

        layer_weights = {
            'conv': ConvLayersWeights,
            'fullyconnect': FFLayersWeights
        }

        self.weights = layer_weights


    def training(self, train_set, epochs = 20000, alfa = 0.01):

        tr_set_size = train_set[0].shape[0]
        batch_max_range = 10

        for ep in range(epochs):

            """ 
            a, b = np.random.randint(0, batch_max_range, 2)

            lB = min(a,b)
            hB = max(a,b)

            if lB == hB: hB += 2

            mn = np.random.randint(0, tr_set_size//batch_max_range)

            lB *= mn
            hB *= mn

            img_batch = train_set[0][lB:hB,:,:,:]
            label_batch = train_set[1][lB:hB,:,:]
            """

            img_batch = np.expand_dims(train_set[0][ep,:,:,:],axis=0)
            label_batch = np.expand_dims(train_set[1][ep,:,:],axis=0)

            Y, inValues = self.forwardPass(img_batch)
            G = self.backwardPass(label_batch,Y, inValues)
            W = self.weights
            self.weights = self.update(alfa,W,G)

    def validation(self, valid_set):

        epochs = 2000
        positive = 0

        for ep in range(epochs):

            result, _ = self.forwardPass(np.expand_dims(valid_set[0][ep,:,:,:],axis=0))

            """
            print(result.shape,end='resShape\n')
            print(result, end=' res\n')
            print(valid_set[1][ep,:,:].shape, end='validShape\n')
            print(valid_set[1][ep,:,:], end=' validShape\n')

            print(np.argmax(result),end='\n\n')
            print(np.argmax(valid_set[1][ep,:,:]))
            """

            if(np.argmax(result[0]) == np.argmax(valid_set[1][ep,:,:])):
                positive += 1

        return positive/epochs


    def forwardPass(self, X):
        """
            :param X:   batch of images (Input value of entire convolutional neural network)
                        image.shape = (m,i,i,c) - c is number of channels
                        for current task, first input c = 1 (grayscale)
                        example: for RGB c = 3
                        m - batch size
                        X.shape M x I x I x C

            :return :   touple(Z, inValues)
                        Z - estimated probability of every class
                        Z.shape M x K x 1


        """

        W = self.weights

        inValues = {
            'conv': [],
            'fullyconnect': [],
            'mask' : [],
            'pooling' : [],
            'flatten' : [],
            'sigmoid' : [],
            'relu' : []
        }

        """
            Current structure:
            Conv -> Relu -> Conv -> Relu -> MaxPooling -> Conv -> Relu -> Conv -> Relu -> MaxPooling -> Flattening -> 
            -> sigmoid -> Fully connected -> sigmoid ->Fully connected -> softmax
        """



        inValues['conv'].append(X)
        Z = self.convolution_layer(X, W['conv'][0]);z = Z

        inValues['relu'].append(z)
        Z = self.relu(z);z =Z

        inValues['conv'].append(z)
        Z = self.convolution_layer(z, W['conv'][1]);z = Z


        inValues['relu'].append(z)
        Z = self.relu(z);z =Z


        inValues['pooling'].append(z)
        Z, mask = self.max_pooling(z);z = Z
        inValues['mask'].append(mask)



        inValues['conv'].append(z)
        Z = self.convolution_layer(z, W['conv'][2]);z = Z

        inValues['relu'].append(z)
        Z = self.relu(z);z = Z

        inValues['conv'].append(z)
        Z = self.convolution_layer(z, W['conv'][3]);z = Z

        inValues['relu'].append(z)
        Z = self.relu(z);z = Z

        inValues['pooling'].append(z)
        Z, mask = self.max_pooling(z);z = Z
        inValues['mask'].append(mask)


        inValues['flatten'].append(z)
        Z = self.flattening(z);z = Z


        inValues['sigmoid'].append(z)
        Z = self.sigmoid(z); z = Z


        inValues['fullyconnect'].append(z)
        Z = self.fullyConnected_layer(z, W['fullyconnect'][0]); z = Z


        #dropout here later

        inValues['sigmoid'].append(z)
        Z = self.sigmoid(z); z = Z


        inValues['fullyconnect'].append(z)
        Z = self.fullyConnected_layer(z, W['fullyconnect'][1]);z = Z


        Z = self.softmax(z)


        return Z, inValues

    def backwardPass(self, y, Y, inValues):

        """

        :param Y: estimated probability of all K classes
                    ( Y.shape = M x K x 1 )
        :param y: True labels for current
                    M x K x 1
        :param inValues: Dictionary with input values of conv/ff layers
                         example: inValues['conv'][1] - Values encountered during feedForward on input of Conv layer with index 1
        :return:  Gradient of weights in respect to L
        """

        np.set_printoptions(suppress=True)
        W = self.weights

        G = {
            'conv' : [],
            'fullyconnect' : []
        }


        Z = self.softmax_backward(Y, y); z = Z



        Z, dW, dB = self.fullyConnected_layer_backward(z, W['fullyconnect'][1],inValues['fullyconnect'][1]);z = Z
        weight = {
            'W': dW,
            'B': dB
        }
        G['fullyconnect'].append(weight)



        Z = self.sigmoid_deriv(z, inValues['sigmoid'][1]); z = Z

        Z, dW, dB = self.fullyConnected_layer_backward(z, W['fullyconnect'][0],inValues['fullyconnect'][0]);z = Z;
        weight = {
            'W': dW,
            'B': dB
        }
        G['fullyconnect'].append(weight)


        Z = self.sigmoid_deriv(z, inValues['sigmoid'][0]);z=Z


        Z = self.flattening_backward(z, inValues['flatten'][0]); z = Z


        Z = self.max_pooling_backward(z,inValues['mask'][1]); z = Z


        Z = z * self.relu(inValues['relu'][3], deriv=True); z = Z

        Z, dW, dB = self.convolution_layer_backward(z, W['conv'][3],inValues['conv'][3]); z = Z
        weight = {
            'W': dW,
            'B': dB
        }
        G['conv'].append(weight)

        Z = z * self.relu(inValues['relu'][2], deriv=True);z = Z


        Z, dW, dB = self.convolution_layer_backward(z, W['conv'][2],inValues['conv'][2]); z = Z
        weight = {
            'W': dW,
            'B': dB
        }
        G['conv'].append(weight)


        Z = self.max_pooling_backward(z,inValues['mask'][0]);z = Z


        Z = z * self.relu(inValues['relu'][1], deriv=True);z = Z

        Z, dW, dB = self.convolution_layer_backward(z, W['conv'][1],inValues['conv'][1]); z = Z
        weight = {
            'W': dW,
            'B': dB
        }
        G['conv'].append(weight)

        Z = z * self.relu(inValues['relu'][0], deriv=True);z = Z

        Z, dW, dB = self.convolution_layer_backward(z, W['conv'][0],inValues['conv'][0]); z = Z
        weight = {
            'W': dW,
            'B': dB
        }
        G['conv'].append(weight)

        G['conv'].reverse()
        G['fullyconnect'].reverse()

        return G

    def update(self, alfa, W, G):

        W['fullyconnect'][0]['W'] -= alfa * np.sum(G['fullyconnect'][0]['W'],axis=0)
        W['fullyconnect'][1]['W'] -= alfa * np.sum(G['fullyconnect'][1]['W'],axis=0)
        W['fullyconnect'][0]['B'] -= alfa * np.sum(G['fullyconnect'][0]['B'],axis=0)
        W['fullyconnect'][1]['B'] -= alfa * np.sum(G['fullyconnect'][1]['B'],axis=0)

        W['conv'][0]['W'] -= alfa * np.sum(G['conv'][0]['W'],axis=0)
        W['conv'][1]['W'] -= alfa * np.sum(G['conv'][1]['W'],axis=0)
        W['conv'][2]['W'] -= alfa * np.sum(G['conv'][2]['W'],axis=0)
        W['conv'][3]['W'] -= alfa * np.sum(G['conv'][3]['W'],axis=0)
        W['conv'][0]['B'] -= alfa * np.sum(G['conv'][0]['B'],axis=0)
        W['conv'][1]['B'] -= alfa * np.sum(G['conv'][1]['B'],axis=0)
        W['conv'][2]['B'] -= alfa * np.sum(G['conv'][2]['B'],axis=0)
        W['conv'][3]['B'] -= alfa * np.sum(G['conv'][3]['B'],axis=0)

        return W



    def convolution_layer(self, X, W):

        """
        Convolution performing with 0 padding (valid) and stride 1
        Using Fast fourier transform (whatever it is xD)

        :param X: batch of images
                    X.shape M x J x J x C
        :param W: Dictionary of layer weights:
                    W['W'] = filters (F x F x C x K)
                    W['B'] = biases ( K x 1 )
        :return: batch of convoluted images
        """

        Bias = W['B']
        W = W['W']

        F,_, C, K = W.shape
        M, J, _, _ = X.shape

        N = J - F + 1

        """
            F - Filter size
            C - Number of channels
            K - Number of filters
            M - number of training examples (batches)
            J - image size
        """

        y = np.zeros((M,N,N,K))

        W = np.flip(W, axis=0)  #flipping filters for convolution not coleration process
        W = np.flip(W, axis=1)


        for m in range(M):      #every batch
            for k in range(K):      #every filter
                for c in range(C):      #every channel of input
                    patch = X[m, :, :, c]
                    filter = W[:,:,c,k]
                    convolved = self.convolve2d( patch, filter,border_mode='valid')
                    y[m,:,:,k] += convolved

        y = y + Bias[0,:]

        return y

    def convolution_layer_backward(self, dZ, W, X):
        """

        :param dZ:  gradient of previous layer
                    dZ.shape M x N x N x K
        :param W:   Dictionary of layer weights:
                    W['W'] = filters (F x F x C x K)
                    W['B'] = biases ( K x 1 )
        :param X:   Values encountered on input during feedForward
        :return:    touple(dX, dW, dB)

        """

        W = W['W']

        M, N, _, K = dZ.shape
        F, _, C, _ = W.shape
        _, J, _, _ = X.shape

        """
            N - Derivative of previous layer size ( dZ size)
            F - Filter size
            C - Number of channels
            K - Number of filters
            M - number of training examples (batches)
            J - image size
        """

        #dB = np.sum(dZ, axis=(1,2), keepdims=True)
        dB = np.expand_dims(np.sum(dZ, axis=( 1, 2)),axis=-1)


        dW = np.zeros_like(W)
        X_bcast = np.expand_dims(X, axis=-1)
        dZ_bcast = np.expand_dims(dZ, axis=-2)

        for a in range(F):
            for b in range(F):
                filter_range_on_x = J - F + 1
                x = X_bcast[:, a : filter_range_on_x + a, b : filter_range_on_x + b,:,:]
                patch = np.sum(dZ_bcast * x, axis=(0,1,2))
                dW[a,b,:,:] = (1/M) * patch

        dX = np.zeros_like(X, dtype=float)
        dZ_pad = F - 1
        dZ_padded = np.pad(dZ,((0,0),(dZ_pad,dZ_pad),(dZ_pad,dZ_pad),(0,0)), 'constant', constant_values=0)

        for m in range(M):
            for k in range(K):
                for c in range(C):
                    dz = dZ_padded[m,:,:,k]
                    w = W[:,:,c,k]
                    conv_res = self.convolve2d(dz,w)

                    dX[m, :, :, c] += conv_res

        return dX, dW, dB

    def relu(self,X, deriv = False):
        if deriv:
            return X > 0
        return np.multiply(X, X > 0)

    def max_pooling(self,X):

        J = X.shape[1]
        newJ = J // 2
        flag = False # for special case when J is odd

        # this will allow us to easy sum over specified axises, if statement for checking do we cut anything(*)
        if J & 1:
            X = X[:, :-1, :-1, :]
            flag = True


        patch = X.reshape(X.shape[0], newJ, 2, newJ, 2, X.shape[3])

        out = patch.max(axis=2).max(axis=3)
        # isClose is gonna repeat argmax for mask creating (mask used in backprop
        mask = np.isclose(X,np.repeat(np.repeat(out,2,axis=1),2,axis=2)).astype(int)

        if flag:
            mask = np.pad(mask, ((0, 0), (0, 1), (0, 1), (0, 0)), 'constant', constant_values=0)


        return out, mask

    def max_pooling_backward(self,X, mask):
        x = np.repeat(np.repeat(X, 2, axis=1), 2, axis=2)

        if x.shape[1] < mask.shape[1]:
            x = np.pad(x, ((0, 0),(0, 1), (0, 1), (0, 0)), 'constant', constant_values=0)

        return mask*x

    def flattening(self,Z):
        M, X, _, C = Z.shape
        res = Z.reshape((M,X*X*C))
        res = np.expand_dims(res, axis=-1)
        return res

    def flattening_backward(self, X, Z):
        return X.reshape(Z.shape)


    def fullyConnected_layer(self,X, W):
        """

        :param X: input for neural network
                  X.shape M x J x 1
        :param W: Dictionary of layer weights:
                    W['W'] = Weights ( K x J)
                    W['B'] = biases ( K x 1 )
        :return:  Layer output
                  shape M x K x 1
        """

        Bias = W['B']
        W = W['W']

        M = X.shape[0]
        K = Bias.shape[0]

        Z = np.zeros((M,K,1))

        for m in range(X.shape[0]):
            Z[m,:,:] = np.dot(W, X[m,:,:]) + Bias

        return Z

    def fullyConnected_layer_backward(self, dZ, W , X):
        """
        :param dZ:  gradient of previous layer
                    dZ.shape M x K x 1
        :param W:   Dictionary of layer weights:
                    W['W'] = Weights ( K x J)
                    W['B'] = biases ( K x 1 )
        :param X:   Values encountered on input during feedForward
                    X.shape M x J x 1
        :return:    touple(dX, dW, dB)
        """

        W = W['W']
        M, K, _ = dZ.shape
        _, J, _ = X.shape

        dW = np.zeros((M,K,J))
        dX = np.zeros((M,J,1))

        for m in range(M):
            dW[m] = np.dot(dZ[m,:,:],X[m,:,:].T)
            dX[m] = np.dot(W.T, dZ[m,:,:])

        dB = dZ


        return dX, dW, dB


    def sigmoid(self, X):
        sig = 1 / (1 + np.exp(-X))
        return sig

    def sigmoid_deriv(self, dZ, X):
        """

        :param dZ: previous layer derivative
        :param X:  input Value
        :return:  derivative for current layer sigmoid
        """
        sig = 1 / (1 + np.exp(-X))
        return (sig * (1 - sig)) * dZ

    def softmax(self, w):
        w -= np.mean(w,axis=0,keepdims=True)
        res = np.exp(w)
        divider = np.sum(res,axis=0,keepdims=True)
        return (res/divider) + 1e-8

    def softmax_backward(self, Y, y):
        """

        :param Y: estimated prob
                    M x K x 1
        :param y: label (one hot encoding )
                    M x K x 1
        :return:    derivative of softmax in current layer
        """

        return Y - y




    @staticmethod
    def convolve2d(image, feature, border_mode="valid"):
        image_dim = np.array(image.shape)
        feature_dim = np.array(feature.shape)
        target_dim = image_dim + feature_dim - 1
        fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
        target = np.fft.ifft2(fft_result).real

        if border_mode == "valid":
            # To compute a valid shape, either np.all(x_shape >= y_shape) or
            # np.all(y_shape >= x_shape).
            valid_dim = image_dim - feature_dim + 1
            if np.any(valid_dim < 1):
                valid_dim = feature_dim - image_dim + 1
            start_i = (target_dim - valid_dim) // 2
            end_i = start_i + valid_dim
            target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]

        return target











