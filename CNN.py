import numpy as np



class ConvNN():

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
        pass






    def forwardPass(self, X):
        """
            :param X:   batch of images (Input value of entire convolutional neural network)
                        image.shape = (m,i,i,c) - c is number of channels
                        for current task, first input c = 1 (grayscale)
                        example: for RGB c = 3
                        m - batch size
                        X.shape M x I x I x C

            :return :   touple(y, Y, inValues) where:
                        y - numbers of class with highest estimated probability {0,1,...,K}
                            y.shape M x 1
                        Y - estimated probability of all classes ( shape: K x 1 )
                            Y.shape M x K x 1
                        inValues: Dictionary with input values of conv/ff layers
                            example: inValues['conv'][1] - Values encountered during feedForward on input of Conv layer with index 1

        """

        W = self.weights

        inValues = {
            'conv': [],
            'fullyconnect': []
        }

        inValues['conv'].append(X)
        Z = self.convolution_layer(X, W['conv'][0]);z = Z

        Z = self.relu(z);z =Z

        Z = self.max_pooling(z);z = Z

        inValues['conv'].append(z)
        Z = self.convolution_layer(z, W['conv'][1]);z = Z

        Z = self.relu(z);z = Z

        Z = self.max_pooling(z);z = Z

        Z = self.flattening(z);z = Z

        inValues['fullyconnect'].append(z)
        Z = self.fullyConnected_layer(z, W['fullyconnect'][0]); z = Z

        #dropout here later

        Z = self.sigmoid(z); z = Z

        inValues['fullyconnect'].append(z)
        Z = self.fullyConnected_layer(z, W['fullyconnect'][1]);z = Z

        Z = self.softmax(z)

        return self.classify(Z), Z

    def backwardPass(self, y, Y, inValues):

        """

        :param Y: estimated probability of all K classes
                    ( y.shape = M x K x 1 )
        :param y: True labels for current
        :param inValues: Dictionary with input values of conv/ff layers
                         example: inValues['conv'][1] - Values encountered during feedForward on input of Conv layer with index 1
        :return:  Avarage error over entire batch ???
        """



        W = self.weights
        G = self.weights_grad


        Z = self.softmax_backward(Y, y); z = Z

        Z, dW, dB = self.fullyConnected_layer_backward(z, W['fullyconnect'][1]);z = Z
        G['fullyconnect'][1]['W'], G['fullyconnect'][1]['B'] = dW, dB

        Z = self.sigmoid(z,deriv = True); z = Z

        Z, dW, dB = self.fullyConnected_layer_backward(z, W['fullyconnect'][0],inValues['conv'][1]);z = Z;
        G['fullyconnect'][0]['W'], G['fullyconnect'][0]['B'] = dW, dB

        Z = self.flattening_backward(z); z = Z

        Z = self.max_pooling_backward(z); z = Z

        Z = self.relu(z, deriv=True); z = Z

        Z, dW, dB = self.convolution_layer_backward(z, W['conv'][1],inValues['conv'][1]); z = Z
        G['conv'][1]['W'], G['conv'][1]['B'] = dW, dB

        Z = self.max_pooling_backward(z);z = Z

        Z, dW, dB = self.relu(z, deriv=True);z = Z

        Z, dW, dB = self.convolution_layer_backward(z, W['conv'][0],inValues['conv'][1]); z = Z
        G['conv'][0]['W'], G['conv'][0]['B'] = dW, dB

        pass

    def convolve2d(self,image, feature, border_mode="valid"):
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
        y = y + Bias

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

        dB = np.sum(dZ, axis=(0,1,2), keepdims=True)

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











