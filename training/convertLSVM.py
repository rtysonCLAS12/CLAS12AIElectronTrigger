import numpy as np
import math
from sklearn.datasets import load_svmlight_file

inputFiles=[]
#inputFiles.append('/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/drift_tracks_single.csv')
inputFiles.append('/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/drift_tracks_multiple.csv')

names=[]
#names.append('/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/single')
names.append('/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/denoiseGagik/multiple')

for fileNb in range(len(inputFiles)):
    #LSVM=np.load(inputFile,allow_pickle=True)
    LSVM,y=load_svmlight_file(inputFiles[fileNb])
    print(LSVM.shape)
    print(y.shape)

    #print(LSVM[0])
    #print('1')
    #print(LSVM[1])
    #print('y')
    #print(y)


    #print()

    outFileNb=0

    nFiles=10

    wroteIn=0
    wroteOut=0

    nInFile=50000

    nWriteOut=1000000
    if(LSVM.shape[0]<nWriteOut):
        nWriteOut=LSVM.shape[0]

    if(round(LSVM.shape[0]/2)<nInFile):
        nInFile=round(LSVM.shape[0]/2)

    DCIn=np.zeros((nInFile,6,112))
    DCOut=np.zeros((nInFile,6,112))

    for i in range(nWriteOut):

        if((i%10000)==0):
            print('file '+str(fileNb)+' entry '+str(i)+' from '+str(LSVM.shape[0]))

        if(i!=0):
            if((i%100000)==0):
                np.save(names[fileNb]+'_in_'+str(outFileNb)+'.npy',DCIn)
                np.save(names[fileNb]+'_out_'+str(outFileNb)+'.npy',DCOut)
                print('Writing file: '+str(outFileNb))
                outFileNb=outFileNb+1
                DCIn=np.zeros((50000,6,112))
                DCOut=np.zeros((50000,6,112))
                wroteIn=0
                wroteOut=0

        DCIm=np.zeros((6,112))
        row,cols=LSVM[i].nonzero()
        for col in cols:
            layer=math.floor(col/112)
            wire=col%112
            #print('place '+str(col)+' layer '+str(layer)+' wire '+str(wire)+' val '+str(LSVM[i].todense()[0,col]))
            DCIm[layer,wire]=LSVM[i].todense()[0,col]
        
        if(y[i]==0):
            DCIn[wroteIn]=DCIm
            wroteIn=wroteIn+1
        else:
            DCOut[wroteOut]=DCIm
            wroteOut=wroteOut+1

    np.save(names[fileNb]+'_in_'+str(outFileNb)+'.npy',DCIn)
    np.save(names[fileNb]+'_out_'+str(outFileNb)+'.npy',DCOut)
        
    
    
    #for i in range(len(LSVM)):
    #    
    #    for j in range(1,LSVM.shape[1]):
    #        DCIm
    #            if LVSM[i,0]=0:
        
