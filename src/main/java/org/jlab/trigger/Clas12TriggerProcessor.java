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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
    public void initNetwork(String url){
    	try {
        	network = KerasModelImport.importKerasModelAndWeights(url);
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
	 *             OutputDataStream to which the model results are passed
	 */
    public void processNext(InputDataStream stream,OutputDataStream outStream){
    	INDArray[] array = stream.next();
        INDArray result = network.output(array)[0];
        //applyThreshold(result); just write out raw output
        apply(outStream,result);
    }

     /**
	  *  Do something with the trigger model predictions
	  *  
	  * Arguments:
	  *                     outStream: OutputDataStream to which the model results are passed
	  *
	  * 			results: INDArray containing the model predictions.
	  *
	  */
    public void apply(OutputDataStream outStream,INDArray result) {
	long batchSize=result.shape()[0]/6;

	INDArray resultsOut=Nd4j.zeros(batchSize,7);
    	for(int event=0;event<batchSize;event++) {
	    resultsOut.putScalar(new int[] {event,0},event);
	    for(int sector=0;sector<6;sector++) {
		int entry=event*6+sector;
		resultsOut.putScalar(new int[] {event,sector+1}, result.getFloat(entry,0));
	       
            }
        }
	
	outStream.output(resultsOut);

    }//END apply
    
    public static void main(String[] args){
        Clas12TriggerProcessor processor = new Clas12TriggerProcessor();
        processor.initNetwork("trainedModel_rgc_8nA_inbending.h5");//trainedModel_rgb_50nA_inbending.h5 trained_model.h5
        
        HipoInputDataStream stream = new HipoInputDataStream();
        String fName="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/rg-c/rec_clas_016246.evio.00000.hipo";
        stream.open(fName);
        processor.setThreshold(0.2);
        

	HipoOutputDataStream outStream = new HipoOutputDataStream();
	outStream.createBank();
	String fNameOut="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/rg-c/outTest.hipo";
	outStream.open(fNameOut);

	int NBatches=0;

        //while(stream.hasNext()){
	while(NBatches<1){
            System.out.println("Batch: "+NBatches);
            processor.processNext(stream,outStream);  
            System.out.println("");
            NBatches++;
        }

	outStream.close();
        
    }
}
