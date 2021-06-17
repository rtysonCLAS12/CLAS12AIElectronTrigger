import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import metrics as mt
import math
import seaborn as sns

#shape [N,6,184]
data=np.load("/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/trainingSamples/Sector_signal0_2D.npy")

#DC in first 112 columns
DC=data[:,:,:112,]

#EC in last 72 columns
EC=data[:,:,112:,]

saveDir="plots/"
for i in range(20):
    dc=DC[i]      
    y_axis_labelsDC = ['1', '2','3','4', '5','6']        
    fig=plt.figure(figsize = (20,10))
    axDC=sns.heatmap(dc,cmap='Blues',yticklabels=y_axis_labelsDC, vmin=0, vmax=1)
    axDC.invert_yaxis()
    axDC.set(xlabel="Wire")
    axDC.set(ylabel="Superlayer")
    plt.savefig(saveDir+'DC_'+str(i)+'.png')
    
    ec=EC[i]
    y_axis_labels = ['PCAL U-View', 'PCAL V-View','PCAL W-View','ECIN & ECOUT U-View', 'ECIN & ECOUT V-View','ECIN & ECOUT W-View'] # labels for x-axis
    fig=plt.figure(figsize = (20,10))
    axEC=sns.heatmap(ec,cmap='Blues',yticklabels=y_axis_labels,norm=LogNorm(vmin=0.0001, vmax=0.1))
    axEC.invert_yaxis()
    axEC.set(xlabel="Strip")
    plt.savefig(saveDir+'ECal_'+str(i)+'.png')
 









