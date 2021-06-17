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
from matplotlib.colors import LogNorm
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers as opt 
from tensorflow.keras import metrics as mt
from tensorflow.keras import backend as K
import tensorflow as tf
import math
import time
import seaborn as sns


def nll_loss_fn(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)



t0 = time.time()
saveDir='plots/' #Directory where the plots are printed

#load data separated into positive and negative samples
signal=np.load("data/signal19.npy") 

#load positive sample separated into several files
for i in range(20,29):
    print("signal"+str(i))
    a = np.load("data/signal"+str(i)+".npy")
    signal=np.vstack((signal,a))

print(signal.shape)

#create Y label of 1 for positive sample
signalLab0=np.zeros([len(signal),1])
signalLab1=np.ones([len(signal),1])
signalLab=np.vstack((signalLab1,signalLab0))

bg=np.load("data/bg19.npy")
for i in range(20,29):
    print("bg"+str(i))
    b=np.load("data/bg"+str(i)+".npy")
    bg=np.vstack((bg,b))
    
print(bg.shape)
bg=bg[0:len(signal),:] #balance datasets

#create Y label of 0 for negative sample
bgLab0=np.zeros([len(bg),1])
bgLab1=np.ones([len(bg),1])
bgLab=np.vstack((bgLab0,bgLab1))

#stack both positive and negative samples
X= np.vstack((signal,bg))
Y= np.hstack((signalLab,bgLab))

print(X.shape)

#reshape to add channel (RGB in color images, here just 1 channel)
X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)

#separate into DC and EC features
X_DC=X[:,:,:112]

X_EC=X[:,:,112:]

#load model
model = load_model('trained_model2.h5')


#evaluate prediction rate
t0_inf = time.time()

#apply model to data
y_prob = model.predict({"DC": X_DC, "EC": X_EC},batch_size=128)[:, 0]

t1_inf = time.time()
time_inf=t1_inf-t0_inf

rate=len(Y)/time_inf

print(" Inference time: "+str(time_inf)+"s for "+str(len(Y))+" events.")
print("ie rate of "+str(rate)+" Hz")
print("ie per event "+str(rate/6)+" Hz")
print("ie per record "+str(rate/600)+" Hz \n")

#calculate metrics as function of cut on response
y_testSigCol=Y[:,0] #1 for positive sample, 0 for negative sample
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
for j in range(0,100): #vary cut by 0.01 on response
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

print('Rate: '+str(rate)+' PaS: '+str(bestPuratSen))

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
pyplot.scatter(Thresh, Sen, marker='o', color='blue',label='Efficiency')
pyplot.scatter(Thresh, PS, marker='o', color='darkviolet',label='P*E')
pyplot.ylim(0.825, 1.01)
pyplot.legend(loc='lower center')
pyplot.xlabel('Lower Threshold on Response')
pyplot.ylabel('Metrics')
pyplot.axhline(y = 1.0, color = 'black', linestyle = '--') 
pyplot.axhline(y = 0.995, color = 'grey', linestyle = '--') 
pyplot.title('Metrics vs Response Lower Threshold')
pyplot.savefig(saveDir+'metrics_response.png')

t1 = time.time()
time=t1-t0
time_m=time/60
print("Total time: "+str(time)+"s ie "+str(time_m)+" minutes.")
