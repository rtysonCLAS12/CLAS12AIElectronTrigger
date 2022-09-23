import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import log_loss
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape,UpSampling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers as opt 
from tensorflow.keras import metrics as mt
from tensorflow.keras import backend as K
import tensorflow as tf
import math
import time
import random
import seaborn as sns

def plotDC(dcIm,endName,saveDir):
    y_axis_labelsDC = []
    for SL in range(dcIm.shape[0]):
        y_axis_labelsDC.append(str(SL))
    fig=pyplot.figure(figsize = (10,10))
    axDC=sns.heatmap(dcIm,cmap='Blues',yticklabels=y_axis_labelsDC, vmin=0, vmax=1)
    axDC.invert_yaxis()
    axDC.set(xlabel="Wire")
    axDC.set(ylabel="Superlayer")
    pyplot.savefig(saveDir+'DC_IM'+endName+'.png')
    #plt.show()



def nll_loss_fn(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

t0 = time.time()
saveDir='/home/richardt/public_html/aiTrig/denoiser/'

endName='_wMultiple' 

#print(bg.shape)
X=np.load("/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/single_in_0.npy")
Y= np.load("/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/single_out_0.npy")

for i in range(1,5):
    a=np.load("/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/single_in_"+str(i)+".npy")
    X=np.vstack((X,a))
    a=np.load("/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/single_out_"+str(i)+".npy")
    Y=np.vstack((Y,a))

a=np.load("/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/multiple_in_0.npy")
X=np.vstack((X,a))
a=np.load("/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/multiple_out_0.npy")
Y=np.vstack((Y,a))

print(X.shape)

#allVarsNpy = preprocessing.normalize(allVarsNpy)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=1)

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

y_train=y_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
y_test=y_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

NvarsR_DC = X_train.shape[1] #6
NvarsC_DC = X_train.shape[2] #112

ks=2
st=(1,1)

x=Input(shape=(NvarsR_DC,NvarsC_DC,1),name='in')
h=x
#h=Reshape((NvarsR_DC,NvarsC_DC, 1), input_shape=(NvarsR_DC,NvarsC_DC,))(h)
h=Conv2D(filters=64, kernel_size=ks, strides=(st), padding="same", activation='relu')(h)
h=MaxPool2D(pool_size=(2,2))(h)
h=Conv2D(filters=64, kernel_size=ks, strides=(st), padding="same", activation='relu')(h)
h=UpSampling2D(size=(2, 2))(h)
h=Conv2D(filters=64, kernel_size=ks, strides=(st), padding="same", activation='relu')(h)
h=Conv2D(filters=1, kernel_size=ks, strides=(st), padding="same", activation='relu')(h)
#h=Reshape((6,112,))(h)
model = tf.keras.Model(inputs=x,outputs=h)
model.compile(optimizer='adamax', loss='mse')
model.summary()

#tf.keras.utils.plot_model(model, to_file=saveDir+'model'+endName+'.png',show_shapes=True)

#model = load_model('modelDNN.h5')
history=model.fit(X_train,y_train,epochs=30, validation_data=(X_test, y_test), verbose=2)

fig = pyplot.figure()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left',edgecolor='black')
#pyplot.show()
pyplot.savefig(saveDir+'loss_epoch'+endName+'.png')

model.save("Denoiser_Sector"+endName+".h5")

t0_inf = time.time()

X_test_Denoised = model.predict(X_test,batch_size=128)

X_test_Denoised=X_test_Denoised.reshape(X_test_Denoised.shape[0],6,112,)
X_test=X_test.reshape(X_test.shape[0],6,112,)
y_test=y_test.reshape(y_test.shape[0],6,112,)

t1_inf = time.time()
time_inf=t1_inf-t0_inf

rate=len(y_test)/time_inf

print("\n Inference time: "+str(time_inf)+"s for "+str(len(y_test))+" events.")
print("ie rate of "+str(rate)+" Hz ")
print("ie per event "+str(rate/6)+" Hz \n")


plotDC(y_test[0],endName,saveDir)
plotDC(X_test[0],'_Noised'+endName,saveDir)
plotDC(X_test_Denoised[0],'_Denoised'+endName,saveDir)

t1 = time.time()
time=t1-t0
time_m=time/60
print("Total time: "+str(time)+"s ie "+str(time_m)+" minutes.")
