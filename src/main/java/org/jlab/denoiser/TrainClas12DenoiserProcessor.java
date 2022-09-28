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

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling3D;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.Upsampling1D;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.learning.config.AdaMax;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.dataset.DataSet;


import java.time.Duration;
import java.time.Instant;

/**
 *
 * @authors gavalian, tyson
 */
public class TrainClas12DenoiserProcessor implements DenoiserProcessor {
    
    double inferenceThreshold = 0.5;
    //ComputationGraph network;
    MultiLayerNetwork network;
    
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

	int iterations = 1;
        int seed = 12345;


	//Bunch of options for initialising the network
	//  .iterations(iterations)
	    //  .weightInit(WeightInit.XAVIER)
	    //  .learningRate(0.001)
            //    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	//.updater(Updater.ADAMAX)
	    //.momentum(0.9)
	    //  .regularization(true)
	    //  .l2(0.0005)

	

    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	    .seed(seed)
	    .updater(new AdaMax(0.001))
	    .list()
	    .layer(0, new ConvolutionLayer.Builder()
		   .nIn(1)
		   .nOut(64)
		   .kernelSize(2, 2)
		   .stride(1, 1)
		   .activation(Activation.RELU)
		   .build())
	    .layer(1, new SubsamplingLayer.Builder()
		   .poolingType(SubsamplingLayer.PoolingType.MAX)
		   .kernelSize(2, 2)
		   .stride(1, 1)
		   .build())
	    .layer(2, new ConvolutionLayer.Builder()
		   .kernelSize(2, 2)
		   .stride(1, 1)
		   .nOut(64)
		   .activation(Activation.RELU)
		   .build())
	    /* .layer(3, new ConvolutionLayer.Builder()
		   .kernelSize(2, 2)
		   .stride(1, 1)
		   .nOut(64)
		   .activation(Activation.RELU)
		   .build())*/
	    .layer(3, new Upsampling2D.Builder()
		   //.kernelSize(2, 2)
		   //.stride(1, 1)
		   .build())
	    .layer(4, new ConvolutionLayer.Builder()
		   .kernelSize(2, 2)
		   .stride(1, 1)
		   .nOut(64)
		   .activation(Activation.RELU)
		   .build())
	    .layer(5, new ConvolutionLayer.Builder()
		   .kernelSize(2, 2)
		   .stride(1, 1)
		   .nOut(1)
		   .activation(Activation.RELU)
		   .build())
	    .layer(6, new OutputLayer.Builder()
		   .nOut(1)
		   .lossFunction(LossFunctions.LossFunction.MSE)
		   .activation(Activation.RELU)
		   .build())
	    .setInputType(InputType.convolutionalFlat(6, 112,1))  //convolutionalFlat
	    //.backprop(true)
	    //.pretrain(false)
	    .build();
        network = new MultiLayerNetwork(conf);
        network.init();
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
        INDArray result = network.output(array[2]);
        //applyThreshold(result);
        //stream.apply(result);
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
	    INDArray result = network.output(array[2].reshape(600,1,6,112));
	    Instant end = Instant.now();
	    double timeElapsed = Duration.between(start, end).toNanos()*1e-9; 
	    double rate=array[2].shape()[0]/timeElapsed;
	    avRate+=rate;
	    //System.out.println("Results shape: "+result[0].shape()[0]+"x"+result[0].shape()[1]+"x"+result[0].shape()[2]);
	    if(n==0){output[3]=result;}
	}
	avRate=avRate/100;

	System.out.println("Average Rate: "+avRate+" Hz.");

	
	return output;
    }

    public void train(InputDataStream stream){
	INDArray[] array = stream.next();
	//!! array[0] has noiseless tracks, [1] only noise, [2] both
	DataSet dataset = new DataSet(array[2].reshape(6000,1,6,112),array[0].reshape(6000,1,6,112));

	int epochs=30;
	
        for( int i=0; i<epochs; i++ ) {
	    
            network.fit(dataset);
	    System.out.println("training epoch: "+i);
        }
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
        TrainClas12DenoiserProcessor processor = new TrainClas12DenoiserProcessor();
        processor.initNetwork();

	
	HipoInputDataStream streamT = new HipoInputDataStream();
        String fName="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/out_clas_005038.evio.00105-00109.hipo";
	String fNameNoise="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/background/rga_fall2018/tor-1.00_sol-1.00/55nA_10604MeV/10k/00001.hipo";
        streamT.open(fName,fNameNoise);
	streamT.setBatch(1000);
	processor.train(streamT);
        
        HipoInputDataStream stream = new HipoInputDataStream();
        //String fName="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/out_clas_005038.evio.00105-00109.hipo";
	//String fNameNoise="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/background/rga_fall2018/tor-1.00_sol-1.00/55nA_10604MeV/10k/00001.hipo";
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
