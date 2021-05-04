# CLAS12AIElectronTrigger

This repository contains the java implementation of a deep learning approach to a Level 3 Electron trigger for CLAS12.

## CPU or GPU Usage

To use deeplearning4j or ND4J with a CPU you need to have in pom.xml:

<dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-core</artifactId>
      <version>1.0.0-beta7</version>
 </dependency>
 <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>nd4j-native-platform</artifactId>
      <version>1.0.0-beta7</version>
  </dependency>
 </dependencies>

To use a GPU you can change the artifactID to deeplearning4j-cuda-10.2 for the deeplearning4j library, and to nd4j-cuda-10.2 for the ND4J library. 

## Clas12AIElTrigger.java

Interface with the basic required methods to load the trained neural network, parse data into the required format as input for the network, and to return the network predictions on the parsed data.

## Clas12AIElTriggerHipo.java

Implements Clas12AIElTrigger when reading from a .hipo file (see [Hipo Libaray](https://github.com/gavalian/hipo) ). Some further methods were added to test the neural network performance and compare it to the CLAS12 Electron trigger.

## Tester.java

Class to test the Clas12AIElTriggerHipo implementation, by plotting:
- the prediction rate as a function of batch size
- the classifier response, the accuracy
- the classifier efficiency and purity as a function of the lower threshold on the response
- the purity and efficiency as a function of momentum, with a comparison to the CLAS12 Trigger purity