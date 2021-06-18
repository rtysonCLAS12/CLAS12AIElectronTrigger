# CLAS12AIElectronTrigger

This repository contains the java implementation of a deep learning approach to a Level 3 Electron trigger for CLAS12 and the python scripts requried to train it.

## InputDataStream

Interface used to parse data on which to call the neural network trigger. This is implemented for Hipo files (see [Hipo Library](https://github.com/gavalian/hipo)) in HipoInputDataStream.java. These require the TimeBasedTrkg::TBHits and ECAL::hits banks.

Summary of methods:
- open(String url); opens the data source located at a given url.
- setBatch(int size); sets the number of events to be read in one go. Note at the moment this assumes that each event has 6 entries, one for each sector. The predictions are then made on batches of size size*6.
- hasNext(); returns a boolean if the data stream can still provide more events.
- next(); processes each batch of events.
- apply(INDArray results); uses the network predictions stored in results to apply the trigger decision.

## TriggerProcessor

Interface used to apply the neural network trigger to data parsed from an InputDataStream. The trigger processor is implemented in Clas12TriggerProcessor with an example of its usage in main.

Summary of methods:
- setThreshold(doubleThreshold); sets the desired threshold on the response above which an event is classed as a trigger event.
- initNetwork(); initialises the neural network from the saved weights in trained_model.h5
- processNext(InputDataStream stream); applies the neural network to the inputted data stream. The output of the network is then rounded based on the response and passed back to the stream.


## Tester.java

Class to test the trigger classifier by plotting:
- the prediction rate as a function of batch size
- the classifier response
- the classifier efficiency and purity and accuracy as a function of the threshold on the response
- the purity and efficiency as a function of momentum, with a comparison to the CLAS12 Trigger purity. Note: at the moment the momentum assigned to a sector is based on one of the tracks in the sector, but there can be more than one track per sector.

To run the tester you'll need to change the location of the file specified in the data member fName declared on [line 30](https://github.com/rtysonCLAS12/CLAS12AIElectronTrigger/blob/master/src/org/jlab/trigger/Tester.java#L30). These Hipo files require the TimeBasedTrkg::TBHits, ECAL::hits, REC::Track, REC::Particle and RUN::config banks.

The tester class can also save the DataSet produced on this hipo file. This will be separated into "signal" and "background" files containing the positive and negative sample elements respectively as this makes life easier for training and testing. These files are in the NumPy .npy format to be read in by the python scripts contained in the training directory.

## Training

Directory containing python scripts to plot the ECAL and DC "images" from .npy files as produced by the Tester class (dataVisualisation.npy), along with training (train.py) and testing (test.py) the neural network trigger. The recommended architecture is found in the train.py, the test.py class offers a less thorough test of the classifier than what is provided in the above Tester class. 

Depencencies:
- seaborn
- numpy
- math
- matplotlib
- tensorflow
- time

## CPU or GPU Usage

The neural network implementation relies on the ND4J and Deeplearning4j libraries, which are included in the java project using maven in pom.xml. By default these are set up to use a CPU.

To use a GPU you need to change the artifactId for the deeplearning4j and nd4j dependicies in the pom.xml file. Currently, the lines with "deeplearning4j-cuda-10.2" and "nd4j-cuda-10.2" are commented out but you can switch this out with the current artifactIds for both to switch from CPU to GPU usage. 
