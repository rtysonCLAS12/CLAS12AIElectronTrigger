import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

#shape [N,6,184]
data=np.load("data/positive_0.npy")

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
 









