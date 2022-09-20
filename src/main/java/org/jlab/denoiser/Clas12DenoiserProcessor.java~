/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.denoiser;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.time.Duration;
import java.time.Instant;

/**
 *
 * @authors gavalian, tyson
 */
public class Clas12DenoiserProcessor implements DenoiserProcessor {
    
    double inferenceThreshold = 0.5;
    ComputationGraph network;
    
    /**
	 *  Sets the threshold on the response above which to keep a denoised hit
	 *  
	 * Arguments:
	 *  		threshold: threshold above which to keep a denoised hit
	 *  
	 */
    public void setThreshold(double threshold){ inferenceThreshold = threshold; }
    
    
    /**
	 *  Loads the network from the saved weights.
	 *  
	 */
    public void initNetwork(){
    	try {
        	network = KerasModelImport.importKerasModelAndWeights("denoiser.h5");
        } catch (IOException e) {
        	System.out.println("IO Exception");
        	e.printStackTrace();
        } catch (InvalidKerasConfigurationException e) {
        	System.out.println("Invalid Keras Config");
        	e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
        	System.out.println("Unsupported Keras Config");
        	e.printStackTrace();
        }
    }
    
    /**
	 *  Processes each batch of events.
	 *  
	 *  Arguments:
	 *  		InputDataStream from which the data is passed to the model.
	 *  
	 */
    public void processNext(InputDataStream stream){
    	INDArray[] array = stream.next();
	//!! array[0] has noiseless tracks, [1] only noise, [2] both
        INDArray[] result = network.output(array[2]);
        //applyThreshold(result);
        stream.apply(result);
    }

    /**
	 *  Processes each batch of events 100 times to get average spead.
	 *  
	 *  Arguments:
	 *  		InputDataStream from which the data is passed to the model.
	 *  
	 */
    public INDArray[] processNext_100times(InputDataStream stream){
    	INDArray[] array = stream.next();
	//!! array[0] has noiseless tracks, [1] only noise, [2] both
	
	double avRate=0;

	INDArray[] output=new INDArray[4];
    	output[0]=array[0];
    	output[1]=array[1];
	output[2]=array[2];

	for(int n=0;n<100;n++){
	    Instant start = Instant.now();
	    INDArray[] result = network.output(array[2]);
	    Instant end = Instant.now();
	    double timeElapsed = Duration.between(start, end).toNanos()*1e-9; 
	    double rate=array[2].shape()[0]/timeElapsed;
	    avRate+=rate;
	    //System.out.println("Results shape: "+result[0].shape()[0]+"x"+result[0].shape()[1]+"x"+result[0].shape()[2]);
	    if(n==0){output[3]=result[0];}
	}
	avRate=avRate/100;

	System.out.println("Average Rate: "+avRate+" Hz.");

	
	return output;
    }

    /*
     * Function to save your dataset to a desired location.
     * 
     * Arguments:
     * 			Data: ND4J MultiDataSet containing the DC features
     * 			loc: string to the output directory
     * 			endName: string set at the end of the file name for example to save multiple files
     */
    public static void Save(INDArray Data, String loc,String endName) {
	
	//Write files to desired location
	File fileSignal = new File(loc+endName+".npy");
	try {
	    Nd4j.writeAsNumpy(Data,fileSignal);
	} catch (IOException e) {
	    System.out.println("Could not write file");
	}
    }

    
    public static void main(String[] args){
        Clas12DenoiserProcessor processor = new Clas12DenoiserProcessor();
        processor.initNetwork();
        
        HipoInputDataStream stream = new HipoInputDataStream();
        String fName="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/out_clas_005038.evio.00105-00109.hipo";
	String fNameNoise="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/background/rga_fall2018/tor-1.00_sol-1.00/55nA_10604MeV/10k/00001.hipo";
        stream.open(fName,fNameNoise);
        //processor.setThreshold(0.2);
        int NBatches=0;
        //while(stream.hasNext() && stream.hasNextNoise()){
	
	while(NBatches<2){
            System.out.println("Batch: "+NBatches);
            INDArray[] out=processor.processNext_100times(stream);//assumes batch size 100 
	    //processor.processNext(stream); 
           
	    if(NBatches==0){
		String fNameOut="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/background/rga_fall2018/javaOutput/55nA_";
		Save(out[0],fNameOut,"track");
		Save(out[1],fNameOut,"noise");
		Save(out[2],fNameOut,"wNoise");
		Save(out[3],fNameOut,"Denoised");
		
	    }

            NBatches++;
        }

	
	
        
    }

   
}
