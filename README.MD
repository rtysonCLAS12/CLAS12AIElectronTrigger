# CLAS12AIElectronTrigger

This repository contains the java implementation of a deep learning approach to a Level 3 Electron trigger for CLAS12.

## CPU or GPU Usage

The neural network implementation relies on the ND4J and Deeplearning4j libraries, which are included in the java project using maven in pom.xml. By default these are set up to use a CPU.

To use a GPU you need to change the artifactId for the deeplearning4j and nd4j dependicies in the pom.xml file. Currently, the lines with "deeplearning4j-cuda-10.2" and "nd4j-cuda-10.2" are commented out but you can switch this out with the current artifactIds for both to switch from CPU to GPU usage. 

## Clas12AIElTrigger.java

Interface with the basic required methods to load the trained neural network, parse data into the required format as input for the network, and to return the network predictions on the parsed data.

## Clas12AIElTriggerHipo.java

Implements Clas12AIElTrigger when reading from a .hipo file (see [Hipo Library](https://github.com/gavalian/hipo) ). Some further methods were added to test the neural network performance and compare it to the CLAS12 Electron trigger.

## Tester.java

Class to test the Clas12AIElTriggerHipo implementation, by plotting:
- the prediction rate as a function of batch size
- the classifier response, the accuracy
- the classifier efficiency and purity as a function of the lower threshold on the response
- the purity and efficiency as a function of momentum, with a comparison to the CLAS12 Trigger purity

NB: You need to change the file path defined in fName to point to the correct location. 