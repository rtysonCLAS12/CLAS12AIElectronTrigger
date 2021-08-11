/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.trigger;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @authors gavalian, tyson
 */
public class Clas12TriggerProcessor implements TriggerProcessor {
    
    double inferenceThreshold = 0.5;
    ComputationGraph network;
    
    /**
	 *  Sets the threshold on the response above which to call the trigger
	 *  
	 * Arguments:
	 *  		threshold: threshold above which to call the trigger
	 *  
	 */
    public void setThreshold(double threshold){ inferenceThreshold = threshold; }
    
    /**
	 *  Converts response array to array of 0 and 1 if the response is above the threshold.
	 *  
	 * Arguments:
	 *  		results: array containing the raw output from the model which is then rounded to 0 or 1.
	 *  
	 */
    public void applyThreshold(INDArray result) {
    	int NPreds=(int) result.shape()[0];
    	for(int i=0;i<NPreds;i+=1) {
        	if(result.getFloat(i,0)>inferenceThreshold) {
            	result.putScalar(new int[] {i,0}, 1);
            } else {
            	result.putScalar(new int[] {i,0}, 0);
            }
        }
	}
    
    /**
	 *  Loads the network from the saved weights.
	 *  
	 */
    public void initNetwork(){
    	try {
        	network = KerasModelImport.importKerasModelAndWeights("trained_model.h5");
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
        INDArray result = network.output(array)[0];
        applyThreshold(result);
        stream.apply(result);
    }
    
    public static void main(String[] args){
        Clas12TriggerProcessor processor = new Clas12TriggerProcessor();
        processor.initNetwork();
        
        HipoInputDataStream stream = new HipoInputDataStream();
        String fName="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/out_clas_005038.evio.00105-00109.hipo";
        stream.open(fName);
        processor.setThreshold(0.2);
        int NBatches=0;
        while(stream.hasNext()){
            System.out.println("Batch: "+NBatches);
            processor.processNext(stream);  
            System.out.println("");
            NBatches++;
        }
        
    }
}
