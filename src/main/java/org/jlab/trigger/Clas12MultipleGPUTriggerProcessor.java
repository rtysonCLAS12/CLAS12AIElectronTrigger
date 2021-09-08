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
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.jlab.groot.data.GraphErrors;
import org.jlab.groot.ui.TCanvas;
import java.time.Duration;
import java.time.Instant;

/**
 *
 * @authors gavalian, tyson
 */
public class Clas12MultipleGPUTriggerProcessor implements TriggerProcessor {
    
    double inferenceThreshold = 0.5;
    int NWorkers=1;
    ComputationGraph network;
    ParallelInference pi;
    
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
		 pi = new ParallelInference.Builder(network)
            // BATCHED mode is kind of optimization: if number of incoming requests is too high - PI will be batching individual queries into single batch. If number of requests will be low - queries will be processed without batching
		.inferenceMode(InferenceMode.SEQUENTIAL)//SEQUENTIAL,BATCHED

            // max size of batch for BATCHED mode. you should set this value with respect to your environment (i.e. gpu memory amounts)
		     //.batchLimit(BatchSize)

            // set this value to number of available computational devices, either CPUs or GPUs
		.workers(NWorkers)
		
		.build();
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
        INDArray result = pi.output(array)[0];
        applyThreshold(result);
        stream.apply(result);
    }
    
    public static void main(String[] args){
	
	TCanvas canvasRates = new TCanvas("Event Rate",800,500);
	canvasRates.setDefaultCloseOperation(TCanvas.EXIT_ON_CLOSE);
	canvasRates.setLocationRelativeTo(null);
	canvasRates.setVisible(true);
	
	try {
	    int maxWorkers=4;
	    ComputationGraph network = KerasModelImport.importKerasModelAndWeights("trained_model.h5");
	    for(int i=1; i<maxWorkers;i++){
		Clas12MultipleGPUTriggerProcessor processor = new Clas12MultipleGPUTriggerProcessor();
	    
		ParallelInference pi = new ParallelInference.Builder(network)
		    .inferenceMode(InferenceMode.SEQUENTIAL)
		    .workers(i)
		    .build();
		System.out.println("Nb of workers: "+i);

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
		    processor.setThreshold(0.2);
		    double avRate=0;
		    INDArray[] array = stream.next();
		    for(int n=0;n<100;n++){
			Instant start = Instant.now();
			INDArray result = pi.output(array)[0];
			Instant end = Instant.now();
			double timeElapsed = Duration.between(start, end).toNanos()*1e-9; 
			double rate=BatchSize/timeElapsed;
			System.out.println("Batch: "+n+" rate: "+rate+" time: "+timeElapsed);
			avRate+=rate;
		    }
		    avRate=avRate/100;
		    gRates.addPoint(BatchSize, avRate, 0, 0);
		}
		gRates.setTitle("Event Rate vs Batch Size with "+i+" GPU");
		gRates.setTitleX("Batch Size (in events ie 6 inference per event)");
		gRates.setTitleY("Event Rate [Hz]");
		gRates.setLineColor(i+1);
		gRates.setMarkerColor(i+1);
		if(i==1){
		    canvasRates.draw(gRates,"AP");
		} else{
		    canvasRates.draw(gRates,"sameAP");
		}
	    
	    }
	    canvasRates.getCanvas().getPad(0).setLegend(true);
	    canvasRates.getCanvas().getPad(0).setLegendPosition(400, 250);
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
