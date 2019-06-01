import numpy as np
import pickle
import CNN
from matplotlib import pyplot as plt

def myimshow(I, **kwargs):
    # utility function to show image
    plt.figure();
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)

def extractData():
    with (open('data/train.pkl', 'rb')) as f:
        data = pickle.load(f)

    images = np.array(data[0])
    images = np.expand_dims(images.reshape((images.shape[0], 36, 36)),axis=-1)
    labels_ = np.array(data[1])
    labels = np.zeros((labels_.shape[0], 10))

    for i in range(labels.shape[0]):
        labels[i, labels_[i]] = 1

    training_set = []

    tr_set_num = 55000

    tr_imgs = images[:tr_set_num, :,:,:]
    tr_labels = labels[:tr_set_num]

    training_set.append(tr_imgs)
    training_set.append(tr_labels)

    valid_set = []

    valid_imgs = images[tr_set_num:, :,:]
    valid_labels = labels[tr_set_num:]

    valid_set.append(valid_imgs)
    valid_set.append(valid_labels)

    return training_set,valid_set


tr,vl = extractData()

weight = {
    'W' : np.expand_dims(np.expand_dims(np.array([[0,1,0],[0,1,0],[0,1,0]]),axis=-1),axis=-1),
    'B' : np.ones((1,1))
}

batch = tr[0][:10,:,:]



cnn = CNN.ConvNN()

result = cnn.convolution_layer(batch,weight)

for i in range(result.shape[0]):
    myimshow(result[i,:,:,0])
#plt.show()



b_res = result[:,:,:,:]

back = cnn.convolution_layer_backward(b_res, weight,batch)

out,mask = cnn.max_pooling(batch)

res = cnn.max_pooling_backward(out,mask)

#print(res[0].shape)

img = np.arange(225).reshape(1,15,15,1)
out, mask = cnn.max_pooling(img)

res = cnn.max_pooling_backward(out,mask)

#print(res.shape)
#print(res[0,:,:,0])
















