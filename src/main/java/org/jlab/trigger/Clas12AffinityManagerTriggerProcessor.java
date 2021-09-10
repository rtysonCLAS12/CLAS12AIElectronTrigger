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
import org.jlab.groot.data.GraphErrors;
import org.jlab.groot.ui.TCanvas;
import java.time.Duration;
import java.time.Instant;

/**
 *
 * @authors gavalian, tyson
 */
public class Clas12AffinityManagerTriggerProcessor implements TriggerProcessor {
    
    double inferenceThreshold = 0.5;
    int NWorkers=1;
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
	 *  Sets the number of GPU workers
	 *  
	 * Arguments:
	 *  		nworkers: Number of available GPU workers.
	 *  
	 */
    public void setNWorkers(int nworkers){ NWorkers = nworkers; }
    
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

	Integer deviceID=0;
	if(args.length!=0){
	    deviceID=Integer.parseInt(args[0]);
	}

	Nd4j.getAffinityManager().allowCrossDeviceAccess(false);
	Nd4j.getAffinityManager().unsafeSetDevice(deviceID);
	
	TCanvas canvasRates = new TCanvas("Event Rate with GPU"+deviceID,800,500);
	canvasRates.setDefaultCloseOperation(TCanvas.EXIT_ON_CLOSE);
	canvasRates.setLocationRelativeTo(null);
	canvasRates.setVisible(true);
	
	try {
	    
	     ComputationGraph networkTmp = KerasModelImport.importKerasModelAndWeights("trained_model.h5");

	    GraphErrors gRates= new GraphErrors();
		
	    for(int size=1;size<21;size++) {
		System.out.println("Batch Size: "+size);
		int BatchSize=10*size;
		HipoInputDataStream stream = new HipoInputDataStream();
		//Assumes batch in nb of events ie batch will be of size 6*BatchSize
		stream.setBatch(BatchSize);
		
		//String fName="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/out_clas_005038.evio.00105-00109.hipo";
		String fName= "/tmp/2143411t/data/out_clas_005038.evio.00105-00109.hipo";
		stream.open(fName);
		double avRate=0;
		INDArray[] array = stream.next();
		for(int n=0;n<100;n++){
		    Instant start = Instant.now();
		    INDArray result = networkTmp.output(array)[0];
		    Instant end = Instant.now();
		    double timeElapsed = Duration.between(start, end).toNanos()*1e-9; 
		    double rate=BatchSize/timeElapsed;
		    System.out.println("Batch: "+n+" rate: "+rate+" time: "+timeElapsed);
		    avRate+=rate;
		}
		avRate=avRate/100;
		gRates.addPoint(BatchSize, avRate, 0, 0);
	    }
	    gRates.setTitle("Event Rate vs Batch Size with GPU");
	    gRates.setTitleX("Batch Size (in events ie 6 inference per event)");
	    gRates.setTitleY("Event Rate [Hz]");
	    gRates.setLineColor(2);
	    gRates.setMarkerColor(2);
	    canvasRates.draw(gRates,"AP");
	    
	    canvasRates.getCanvas().getPad(0).setTitle(canvasRates.getTitle());
	    System.out.println("Done!");
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
	
}
