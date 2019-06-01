import numpy as np


def convolve2d(image, feature, border_mode="full"):
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

#print(layer_weights['conv'][0]['b'])

arr = np.flip(np.arange(4).reshape((2,2)))
print(arr)
ar = np.arange(16).reshape((4,4))

res = convolve2d(ar,arr,'valid')

print(ar)

print(res)









