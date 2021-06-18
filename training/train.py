import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from tensorflow.keras.layers import Conv2D, Input, concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers as opt 
from tensorflow.keras import metrics as mt
from tensorflow.keras import backend as K
import tensorflow as tf
import math
import time

def nll_loss_fn(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

t0 = time.time()
saveDir='plots/' #Directory where the plots are printed

#load data separated into positive and negative samples
signal=np.load("data/positive_0.npy")

#load positive sample separated into several files
for i in range(1, 29):
    print("signal"+str(i))
    a = np.load("data/positive"+str(i)+".npy")
    signal=np.vstack((signal,a))

#create Y label of 1 for positive sample
signalLab0=np.zeros([len(signal),1])
signalLab1=np.ones([len(signal),1])
signalLab=np.vstack((signalLab1,signalLab0))

bg=np.load("data/negative_0.npy")
for i in range(1,29):
    print("bg"+str(i))
    b=np.load("data/negative"+str(i)+".npy")
    bg=np.vstack((bg,b))
    
bg=bg[0:len(signal),:]#balance datasets

#create Y label of 0 for negative sample
bgLab0=np.zeros([len(bg),1])
bgLab1=np.ones([len(bg),1])
bgLab=np.vstack((bgLab0,bgLab1))

#stack both positive and negative samples
X= np.vstack((signal,bg))
Y= np.hstack((signalLab,bgLab))

print(X.shape)

#separate at random into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=1)

#reshape to add channel (RGB in color images, here just 1 channel)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

#separate into DC and EC features
X_train_DC=X_train[:,:,:112,:]
X_test_DC=X_test[:,:,:112,:]

X_train_EC=X_train[:,:,112:,:]
X_test_EC=X_test[:,:,112:,:]

print("New array sizes:")
print(X_train_DC.shape)
print(X_train_EC.shape)

NvarsR_DC = X_train_DC.shape[1] #6
NvarsC_DC = X_train_DC.shape[2] #112

NvarsR_EC = X_train_EC.shape[1] #6
NvarsC_EC = X_train_EC.shape[2] #72

#kernel size and stride
ks=2
st=(1,2)

#DC encoder
inputDC = Input(shape=(NvarsR_DC,NvarsC_DC,1), name="DC")
xDC = Conv2D(filters=64, kernel_size=ks, strides=(st), padding="same", activation='relu')(inputDC)
xDC = Conv2D(filters=16, kernel_size=ks, strides=(st), padding="same", activation='relu')(xDC)
xDC_out = Flatten()(xDC)

#ECAL encoder
inputEC = Input(shape=(NvarsR_EC,NvarsC_EC,1), name="EC")
xEC = Conv2D(filters=64, kernel_size=ks, strides=(st), padding="same", activation='relu')(inputEC)
xEC = Conv2D(filters=16, kernel_size=ks, strides=(st), padding="same", activation='relu')(xEC)
xEC_out = Flatten()(xEC)

# Merging subnetworks
x = concatenate([xDC_out, xEC_out])

#Classifier
x = Dense(1000, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(500, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)

recall=0.99
opti=opt.Adam(learning_rate=0.00001) #default adam has lr=0.001, avoids overfitting
model = tf.keras.Model(inputs=[inputDC, inputEC], outputs=x)
model.compile(loss='binary_crossentropy', optimizer=opti, metrics=[mt.PrecisionAtRecall(recall)])
model.summary()

#train model
history=model.fit({"DC": X_train_DC, "EC": X_train_EC},y_train,epochs=30, validation_data=({"DC": X_test_DC, "EC": X_test_EC}, y_test), verbose=2)

#plot loss v epoch to evaluate overfitting
fig = pyplot.figure()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.savefig(saveDir+'loss_epoch.png')


#save model as jason
model_json = model.to_json()  #save just config
with open("trained_model2.json", "w") as f:
    f.write(model_json)

#save model as .h5 loaded in java
model.save("trained_model2.h5")

#evaluate prediction rate
t0_inf = time.time()

#apply model to test data
y_prob = model.predict({"DC": X_test_DC, "EC": X_test_EC},batch_size=128)[:, 0]

t1_inf = time.time()
time_inf=t1_inf-t0_inf

rate=len(y_test)/time_inf

print("\n Inference time: "+str(time_inf)+"s for "+str(len(y_test))+" events.")
print("ie rate of "+str(rate)+" Hz ")
print("ie per event "+str(rate/6)+" Hz \n")

print(y_prob.shape)
#calculate metrics as function of cut on response
y_testSigCol=y_test[:,0] #1 for positive sample, 0 for negative sample
threshB=0
bestAcc=0
bestPur=0
bestSen=0
bestPuratSen=0
bestPS=0
bestThresh=0
proba0=[]
proba1=[]
Thresh=[]
Acc=[]
Pur=[]
Sen=[]
PS=[]
for j in range(100): #vary cut by 0.01 on response
    thresh=threshB+(j*0.01)
    Thresh.append(thresh)
    TP=0
    TN=0
    FN=0
    FP=0
    ind=0
    for i in (y_prob):#loop over predictions
        if y_testSigCol[ind]==0: #check if negative sample
            if j==0:
                proba0.append(i) #fill response plot
            if i <= thresh: #check if below threshold cut
                TN=TN+1
            else:
                FP=FP+1
        else: #positive sample
            if j==0:   
                proba1.append(i) #fill response plot
            if i > thresh: #check if above threshold cut
                TP=TP+1
            else:
                FN=FN+1
        ind=ind+1
    #calculate metrics
    probAcc=(TP+TN)/ind 
    Acc.append(probAcc)
    pur=TP/(TP+FP)
    Pur.append(pur)
    sen=TP/(TP+FN)
    Sen.append(sen)
    ps=pur*sen
    PS.append(ps)
    if sen>0.995: #print metrics for efficiency above 0.995
        fthresh="{:.4f}".format(thresh)
        fpur="{:.4f}".format(pur)
        fsen="{:.4f}".format(sen)
        fprobAcc="{:.4f}".format(probAcc)
        print('Thresh '+str(fthresh)+' Pur '+str(fpur)+' Sen '+str(fsen)+' Acc '+str(fprobAcc))
    if probAcc > bestAcc:
        bestAcc=probAcc
        bestThresh=thresh
    if pur > bestPur:
        bestPur=pur
    if sen > bestSen:
        bestSen=sen
    if ps > bestPS:
        bestPS=ps
    if sen>=0.999:
        if pur>bestPuratSen:
            bestPuratSen=pur
    
print("\n")
print('bestAcc (proba)= '+str(bestAcc)+' at '+str(bestThresh))
print('Best Purity= '+str(bestPur))
print('Best Sensitivity= '+str(bestSen))
print('Best PS= '+str(bestPS))
print('Best Purity at Sensitivy above 0.999 = '+str(bestPuratSen))

#make plots
fig = pyplot.figure()
pyplot.hist(proba1, range=[0,1],bins=100, label='Positive Sample')
pyplot.hist(proba0, range=[0,1],bins=100, edgecolor='red',label='Negative Sample',hatch='/',fill=False)
pyplot.legend(loc='upper center')
pyplot.xlabel('Response')
pyplot.title('Response')
pyplot.yscale('log', nonpositive='clip')
pyplot.savefig(saveDir+'response.png')

fig = pyplot.figure()
pyplot.scatter(Thresh, Acc, marker='o', color='green',label='Accuracy')
pyplot.scatter(Thresh, Pur, marker='o', color='red',label='Purity')
pyplot.scatter(Thresh, Sen, marker='o', color='blue',label='Sensitivity')
pyplot.scatter(Thresh, PS, marker='o', color='darkviolet',label='PS')
pyplot.ylim(0.8, 1.0)
pyplot.legend(loc='lower center')
pyplot.xlabel('Lower Threshold on Response')
pyplot.ylabel('Metrics')
pyplot.title('Metrics vs Response Lower Threshold')
pyplot.savefig(saveDir+'metrics_response.png')

t1 = time.time()
time=t1-t0
time_m=time/60
print("Total time: "+str(time)+"s ie "+str(time_m)+" minutes.")
