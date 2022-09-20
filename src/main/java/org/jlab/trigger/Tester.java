package org.jlab.trigger;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import javax.swing.JFrame;
import java.awt.Dimension;

import org.jlab.groot.data.GraphErrors;
import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;
import org.jlab.groot.graphics.EmbeddedCanvasTabbed;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


public class Tester {
    //Need to change the following to point to the input .hipo file
    static String fName= "/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/out_clas_005038.evio.00105-00109.hipo";
    //static String fName= "/Users/gavalian/Work/DataSpace/evio/clas_003852.evio.981.hipo";
    //static String fName= "infile.hipo";//"/Users/gavalian/Work/DataSpace/level3/out_clas_005038.evio.00105-00109.hipo";

    //Main Canvas, makes life easier for plotting
    static EmbeddedCanvasTabbed masterCanvas = new EmbeddedCanvasTabbed("EvRate","Resp","MetVsResp","MetVsP","MetVsPwC12");
    
    /*
     * Main Function, runs the different relevant tests.
     */
    public static void main(String[] args) {

	//String baseLoc="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/rg-b/"; //rg-m
	String baseLoc="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/background/rga_fall2018/";
	/*String[] dirs= new String[6];
	dirs[0]="015048";
	dirs[1]="015049";
	dirs[2]="015050";
	dirs[3]="015052";
	dirs[4]="015055";
	dirs[5]="015066";*/

	String[] dirs= new String[3];
	dirs[0]=baseLoc+"tor-1.00_sol-1.00/45nA_10604MeV/10k/";
	dirs[1]=baseLoc+"tor-1.00_sol-1.00/50nA_10604MeV/10k/";
	dirs[2]=baseLoc+"tor-1.00_sol-1.00/55nA_10604MeV/10k/";

	
	String outDir=baseLoc+"trainingSamples/";

	String[] outdirs = new String[3];
	outdirs[0]=outDir +"45nA_";
	outdirs[1]=outDir +"50nA_";
	outdirs[2]=outDir +"55nA_";
	
	for (int dir=0;dir<3;dir++){
	    for (int file=1;file<101;file++){//file+=5){

		    String zeros="0000";
		    if(file>9){
			zeros="000";
			if(file==100){zeros="00";}
		    }

		    String fileS=String.valueOf(file);
		    String fileS2=String.valueOf(file+4);

		    //String fName2=baseLoc+dirs[dir]+"/rec_clas_"+dirs[dir]+".evio.000"+fileS+".hipo";
		    
		    //String fName2=baseLoc+dirs[dir]+"/rec_clas_"+dirs[dir]+".evio.000"+fileS+"-000"+fileS2+".hipo";

		    String fName2=dirs[dir]+zeros+fileS+".hipo";

		    int NEvents=10000; //check how many files there are 
		    //Load Data, last two arguments are the minimum and maximum amount of superlayer segments
		    //MultiDataSet Data=Tester.ParseDataForTesting(fName2,NEvents,5,6);
		    MultiDataSet Data=Tester.ParseBackground(fName2,NEvents);
		    String networkLoc="trained_model.h5";
		
		    //String endNameSave=dirs[dir]+"_"+fileS;
		    //You can save the datafiles by uncommenting the next line.
		    //Tester.SaveData(Data, outDir,endNameSave);
		    
		    
		    String endNameSave="background_"+fileS;
		    System.out.printf("writing file "+endNameSave);
		    Tester.SaveBackground(Data, outdirs[dir],endNameSave);

	    }
	}
	

	/*UNCOMMENT FOR TESTING
	
	//Here we measure the EventRate as a function of BatchSize.
	Tester.predRateVsBatchSize(Data,networkLoc);
		
	//plot the metrics used to evaluate the AI Trigger by specifying verbose=true.
	int BatchSize=600;
	int NBatches=(NEvents)/BatchSize;
	boolean verbose=true;
	Tester.Test(Data, NBatches, BatchSize,1,verbose,networkLoc);

	JFrame masterFrame = new JFrame();
	masterFrame.add(masterCanvas);
        masterFrame.pack();
        masterFrame.setMinimumSize(new Dimension(800,500));
        masterFrame.setVisible(true);*/
	
    }
	
    /*
     * Estimates the prediction rate as a function of batch size, for  trials per batch size. At the moment the batch size is taken as the number of predictions with one prediction per sector for each event. The output is plotted as the event rate (ie with 6 predictions per event) vs the batch size.
     *
     * Arguments:
     *           Data: ND4J MultiDataSet containing DC and EC images for both positive and negative sample events, the Labels used to distinguish these and the momentum of the particle associated with the DC track.
     * 	     networkLoc: Neural Network Location
     */
    public static void predRateVsBatchSize(MultiDataSet Data, String networkLoc){
	int bestBatchSize=0;//Prediction Rate changes with batch size
	double bestAvRate=0;
	int TrialPerBatch=100;
	GraphErrors gRates= new GraphErrors();
	GraphErrors gRatesEvent= new GraphErrors();
		
	for(int size=1;size<21;size++) {
	    System.out.println("It: "+size);
	    int BatchSizeT=60*size;
	    double avRate=Test(Data,1, BatchSizeT,TrialPerBatch,false,networkLoc);
	    gRates.addPoint(BatchSizeT, avRate, 0, 0);
	    gRatesEvent.addPoint(BatchSizeT, avRate/6, 0, 0);//6 predictions per event, one for each sector
	    System.out.printf("\trate = : %8d %8.2f\n",BatchSizeT,avRate);
	    if(avRate>bestAvRate) {
		bestAvRate=avRate;
		bestBatchSize=BatchSizeT;
	    }
	}
	System.out.format("Best Event Rate %.3f for Batch Size of %d %n%n",bestAvRate,bestBatchSize);
	//PlotRatesVSBatchSize(gRates, "Prediction");
	PlotRatesVSBatchSize(gRatesEvent, "Event");
    }
	
	
    /*
     * Function to save your dataset to a desired location. The events will be separated into
     * the positive and negative samples as this makes life easier for training and testing purposes.
     * 
     * Arguments:
     * 			Data: ND4J MultiDataSet containing the DC and EC features and the labels
     * 			loc: string to the output directory
     * 			endName: string set at the end of the file name for example to save multiple files
     */
    public static void SaveData(MultiDataSet Data, String loc, String endName) {
	long nPreds=Data.getFeatures()[0].shape()[0];
	//Separate signal and bg files
	INDArray DataArraySignal=Nd4j.zeros(nPreds/2,6,184);
	INDArray DataArrayBg=Nd4j.zeros(nPreds/2,6,184);
	INDArray PsSignal=Nd4j.zeros(nPreds/2);
	INDArray PsBg=Nd4j.zeros(nPreds/2);
	INDArray Labels=Data.getLabels()[0];
	int sigPred=0,bgPred=0;
	for(long i=0; i< nPreds;i++) {
	    //Get DC and EC arrays at index 0 and 1 of Data.getFeatures() and remove channel (ie 4th dimension)
	    INDArray DC=Data.getFeatures()[0].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
	    INDArray EC=Data.getFeatures()[1].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
	    INDArray Ps=Data.getLabels()[1].get(NDArrayIndex.point(i), NDArrayIndex.all());
	    INDArray Both=Nd4j.hstack(DC,EC);//stack DC and EC arrays for output
	    //Fill signal and bg arrays
	    if(Labels.getFloat(i,0)==1) {
		DataArraySignal.get(NDArrayIndex.point(sigPred), NDArrayIndex.all(), NDArrayIndex.all()).assign(Both);
		PsSignal.get(NDArrayIndex.point(sigPred)).assign(Ps);
		sigPred++;
	    } else if(Labels.getFloat(i,0)==0) {
		DataArrayBg.get(NDArrayIndex.point(bgPred), NDArrayIndex.all(), NDArrayIndex.all()).assign(Both);
		PsBg.get(NDArrayIndex.point(bgPred)).assign(Ps);
		bgPred++;
	    }
	}
	//Write files to desired location
	File fileSignal = new File(loc+"positive_"+endName+".npy");
	File fileBg = new File(loc+"negative_"+endName+".npy");
	File fileSignalPs = new File(loc+"Momentum_positive_"+endName+".npy");
	File fileBgPs = new File(loc+"Momentum_negative_"+endName+".npy");
	try {
	    Nd4j.writeAsNumpy(DataArraySignal,fileSignal);
	    Nd4j.writeAsNumpy(DataArrayBg,fileBg);
	    Nd4j.writeAsNumpy(PsSignal,fileSignalPs);
	    Nd4j.writeAsNumpy(PsBg,fileBgPs);
	} catch (IOException e) {
	    System.out.println("Could not write file");
	}
    }

    /*
     * Function to save your background dataset to a desired location.
     * 
     * Arguments:
     * 			Data: ND4J MultiDataSet containing the DC features
     * 			loc: string to the output directory
     * 			endName: string set at the end of the file name for example to save multiple files
     */
    public static void SaveBackground(MultiDataSet Data, String loc, String endName) {
	INDArray DC=Data.getFeatures()[0];
	
	//Write files to desired location
	File fileSignal = new File(loc+endName+".npy");
	try {
	    Nd4j.writeAsNumpy(DC,fileSignal);
	} catch (IOException e) {
	    System.out.println("Could not write file");
	}
    }
	
    /*
     * The Test method loads the neural network and predicts on the inputed data separated into batches.
     * Several plots are then made to evaluate the neural network's performance.
     * An exception is given if the neural network cannot be loaded properly.
     * 
     * Arguments:
     * 			Data: ND4J MultiDataSet containing DC and EC images for both positive and negative sample events,
     * the Labels used to distinguish these and the momentum of the particle associated with the DC track.
     * 			NBatches: The number of batches to separate Data into.
     * 			BatchSize: The size of the batch on which predictions are made.
     * 			TrialPerBatch: Number of times the prediction is repeated on a batch, used to evaluate the average
     * prediction rate.
     * 			verbose: Used to decide if the performance plots should be made or not
     * 			networkLoc: Neural Network Location
     * 
     * Returns:
     * 			The prediction rate averaged over NBatches*TrialPerBatch trials.
     */
    public static double Test(MultiDataSet Data, int NBatches, int BatchSize, int TrialPerBatch, boolean verbose,String networkLoc) {
	int NEvents=NBatches*BatchSize;
	double averageRate=0;
		
	try {
	    ComputationGraph network = KerasModelImport.importKerasModelAndWeights(networkLoc);
	    H1F hRates = new H1F("hRates", 2900, 1000, 30000);
	    H1F hTimes = new H1F("hTimes", 500, 0, 0.1);
			
	    INDArray output=Nd4j.zeros(NEvents,2);
		
	    //loop over NBatches, separate Data into batches
	    for(int batch=0; batch<NBatches;batch++) {
		int batchStart=batch*BatchSize;
		int batchEnd=(batch+1)*BatchSize;
				
		//Separate DC and EC data into batches
		INDArray[] DataBatch=new INDArray[2];
		DataBatch[0]=Data.getFeatures()[0].get(NDArrayIndex.interval(batchStart,batchEnd), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
		DataBatch[1]=Data.getFeatures()[1].get(NDArrayIndex.interval(batchStart,batchEnd), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
				
		//Perform predictions and measure the time it took
		Instant start = Instant.now();
		INDArray outputBatch=network.output(DataBatch)[0];
		Instant end = Instant.now();
				
		//output contains the predictions for all batches
		output.get(NDArrayIndex.interval(batchStart,batchEnd), NDArrayIndex.all()).assign(outputBatch);
				
		//Calculate prediction time and rate
		double timeElapsed = Duration.between(start, end).toNanos()*1e-9; 
		double rate=BatchSize/timeElapsed;
		averageRate+=rate/(NBatches*TrialPerBatch);
		hTimes.fill(timeElapsed);
		hRates.fill(rate/6);
				
		//If more than one trial per batch is specified, the predicition time is measured over all trials
		for(int trial=1;trial<TrialPerBatch;trial++) {
		    start = Instant.now();
		    outputBatch=network.output(DataBatch)[0];
		    end = Instant.now();
					
		    timeElapsed = Duration.between(start, end).toNanos()*1e-9; 
		    rate=BatchSize/timeElapsed;
		    averageRate+=rate/(NBatches*TrialPerBatch);
		    hTimes.fill(timeElapsed);
		    hRates.fill(rate/6);
		}
	    }
			
	    if(verbose) {
		//Plot the prediciton times and rates, these can vary between trials
		if(TrialPerBatch>1) {
		    PlotPredRates(BatchSize, hRates, hTimes);
		}
				
		System.out.format(" Average of %.3f predictions per second or %.3f events per second for %d trials %n%n",averageRate,averageRate/6,(NBatches*TrialPerBatch));
		INDArray Labels=Data.getLabels()[0];
		INDArray P=Data.getLabels()[1];
				
		//Use inbuilt deeplearning4J function to evaluate performance
		Evaluation eval = new Evaluation(2);
		eval.eval(Labels, output);
		System.out.println(eval.stats());
				
		//Plot the performance of the neural network
		PlotResponse(NEvents,output,Labels);
		double bestRespTh=PlotMetricsVsResponse(NEvents,output,Labels);
		boolean wC12TriggerComp=true;
		PlotMetricsVsP(NEvents,output,Labels,P,bestRespTh,wC12TriggerComp);
	    }
			
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

	return (averageRate);
    }//End of Test
	
    /*
     * The PlotMetricsVsP plots the Efficiency and Purity of the neural network as a function
     * of the momentum of the DC track.
     * An option is passed to decide if the CLAS12 Trigger purity for the .hipo file specified in fName
     * should be calculated and plotted with the classifier performance. This involves re-reading this file.
     * 
     * Arguments:
     * 			NEvents: The length of the predictions array.
     * 			predictions: Contains the output of the neural network
     * 			Labels: Contains the true classes for each prediction
     * 			P: The momentum of the DC track
     * 			RespTh: the threshold applied to the classifier response
     * 			wC12TriggerComp: If true, will estimate the CLAS12Trigger purity and plot with the classifier performance
     */
    public static void PlotMetricsVsP(int NEvents,INDArray predictions, INDArray Labels, INDArray P, double RespTh, boolean wC12TriggerComp) {
	INDArray[] C12TriggerAndP=new INDArray[2];
	//if asked for, get the CLAS12 Trigger predictions and truth
	if(wC12TriggerComp) {
	    System.out.println("Re-reading File to Calculate (non AI) C12 Trigger Purity:");
	    C12TriggerAndP= GetFileTriggerAndTruth(NEvents);
	    System.out.format("%n Current (non AI) C12Trigger Purity: %.3f%n%n",calculateCLAS12TriggerPurity(NEvents, C12TriggerAndP[0]));
	}
		
	GraphErrors gEff= new GraphErrors();
	GraphErrors gPur= new GraphErrors();
	GraphErrors gC12Pur= new GraphErrors();
	double inc=0.5;
	//Bin the predictions in momentum and then calculate metrics.
	for(double Pinc=0; Pinc<10; Pinc+=inc) {
	    INDArray binnedPreds = BinInP(NEvents,predictions.dup(), P,Pinc,Pinc+inc);
	    INDArray metrics =getMetrics(NEvents, binnedPreds, Labels, RespTh);
	    double Pur=metrics.getFloat(1,0);
	    double Eff=metrics.getFloat(2,0);
			
	    gPur.addPoint(Pinc+(inc/2), Pur, 0, 0);
	    gEff.addPoint(Pinc+(inc/2), Eff, 0, 0);
			
	    if(wC12TriggerComp) {
		INDArray binnedC12Trigger = BinInP(NEvents,C12TriggerAndP[0].dup(),C12TriggerAndP[1],Pinc,Pinc+inc);
		double C12Pur=calculateCLAS12TriggerPurity(NEvents, binnedC12Trigger);
		gC12Pur.addPoint(Pinc+(inc/2), C12Pur, 0, 0);
	    }
	}//Increment threshold on response
		
	//Make plots
	gEff.setTitle("Efficiency (Blue)");
	gEff.setTitleX("Momentum [GeV]");
	gEff.setTitleY("Metrics");
	gEff.setMarkerColor(4);
	gEff.setMarkerStyle(8);
		
		
	gPur.setTitle("Purity (Red)");
	gPur.setTitleX("Momentum [GeV]");
	gPur.setTitleY("Metrics");
	gPur.setMarkerColor(2);
	gPur.setMarkerStyle(8);

	masterCanvas.getCanvas("MetVsP").getPad(0).setLegend(true);
	masterCanvas.getCanvas("MetVsP").getPad(0).setLegendPosition(400, 300);

	masterCanvas.getCanvas("MetVsP").draw(gEff,"AP");
	masterCanvas.getCanvas("MetVsP").draw(gPur,"sameAP");
	masterCanvas.getCanvas("MetVsP").getPad(0).setTitle("Metrics vs Momentum");
		
	if(wC12TriggerComp) {
	    gC12Pur.setTitle("CLAS12 Trigger Purity (Green)");
	    gC12Pur.setTitleX("Momentum [GeV]");
	    gC12Pur.setTitleY("Metrics");
	    gC12Pur.setMarkerColor(3);
	    gC12Pur.setMarkerStyle(8);

	    masterCanvas.getCanvas("MetVsPwC12").getPad(0).setLegend(true);
	    masterCanvas.getCanvas("MetVsPwC12").getPad(0).setLegendPosition(400, 300);
			
	    masterCanvas.getCanvas("MetVsPwC12").draw(gEff,"AP");
	    masterCanvas.getCanvas("MetVsPwC12").draw(gPur,"sameAP");
	    masterCanvas.getCanvas("MetVsPwC12").draw(gC12Pur,"sameAP");
	    masterCanvas.getCanvas("MetVsPwC12").getPad(0).setTitle("Metrics vs Momentum with CLAS12 Trigger Comparison");
	}
    }//End of PlotMetricsVSP
	
    /*
     * Calculates the CLAS12 Electron Trigger purity given it's predictions
     * 
     * Arguments:
     * 			NEvents: The length of the Trigger array.
     * 			Trigger: NEvents*2 array, the first column is 1 (0) if the C12Trigger has been called (or not)
     * the second column is 1 (0) if an electron is reconstructed in the same sector (or not)
     * 
     * Returns:
     * 			The purity of the CLAS12 Electron Trigger
     */
    public static double calculateCLAS12TriggerPurity(int NEvents, INDArray Trigger) {
	double TP=0, FP=0;
	for(int i=0;i<NEvents;i+=1) {
	    if(Trigger.getFloat(i,0)!=-1) { //Used to bin in P
		if(Trigger.getFloat(i,0)==1) { //Trigger was set in a given sector
		    if(Trigger.getFloat(i,1)==1) { //EB Electron in same sector
			TP++;
		    } else {
			FP++;
		    }
		}// Could add case where trigger isn't set to calculate efficiency
	    }
	}
	double Purity=TP/(TP+FP);
	return Purity;
    }
	
    /*
     * Bins the inputed prediction array in momentum. This binning is done by setting the prediction values to -1
     * if the momentum of the associated DC track is outside of [pMin,pMax[. -1 values are then ignored
     * when calculating the performance metrics.
     * 
     * Arguments:
     * 			NEvents: The length of the predictions array.
     * 			predictions: Contains the output of the neural network
     * 			P: The momentum of the DC track associated with the prediction
     * 			pMin: Start of the bin
     * 			pMax: End of the bin
     * 
     * Returns:
     * 			The predictions binned in P (ie when the momentum is out of bounds the prediction is set to -1).
     */
    public static INDArray  BinInP(int NEvents, INDArray predictions, INDArray P, double pMin, double pMax) {
	for(int i=0;i<NEvents;i+=1) {
	    if(P.getFloat(i,0)<pMin || P.getFloat(i,0)>=pMax) {
		predictions.putScalar(new int[] {i,0}, -1);
	    }
	}
	return predictions;
    }// End of BinInP
	
    /*
     * Calculates the Accuracy, Efficiency and Purity of the neural network as a function for a given
     * threshold on the response.
     * 
     * Arguments:
     * 			NEvents: The length of the predictions array.
     * 			predictions: Contains the output of the neural network
     * 			Labels: Contains the true classes for each prediction
     * 			RespTh: the threshold applied to the classifier response
     * 
     * Returns:
     * 			INDArray which is 3*1. The first row contains the accuracy, the second the purity,
     * the third the efficiency of the neural network
     */
    public static INDArray getMetrics(int NEvents,INDArray predictions, INDArray Labels, double RespTh) {
	INDArray metrics = Nd4j.zeros(3,1);
	double TP=0,FN=0,FP=0,TN=0;
	for(int i=0;i<NEvents;i+=1) {
	    if(predictions.getFloat(i,0)!=-1) {
		if(Labels.getFloat(i,0)==1) {
		    if(predictions.getFloat(i,0)>RespTh) {
			TP++;
		    } else {
			FN++;
		    }//Check model prediction
		} else if(Labels.getFloat(i,0)==0) {
		    if(predictions.getFloat(i,0)>RespTh) {
			FP++;
		    } else {
			TN++;
		    }//Check model prediction
		}//Check true label
	    }//Check that prediction not equals -1, used when binning in P to ignore values
	}//loop over events
	double Acc=(TP+TN)/(TP+TN+FP+FN);
	double Pur=TP/(TP+FP);
	double Eff=TP/(TP+FN);
	metrics.putScalar(new int[] {0,0}, Acc);
	metrics.putScalar(new int[] {1,0}, Pur);
	metrics.putScalar(new int[] {2,0}, Eff);
	return metrics;
    }//End of getMetrics
	
    /*
     * Plots event or prediction rates as a function of BatchSize, for 6 predictions per event, one per sector.
     * 
     * Arguments:
     * 			gRates: GraphError class to be plotted
     * 			predType: String for the titles, recommend using "Event" or "Prediction"
     *
     */
    public static void PlotRatesVSBatchSize(GraphErrors gRates, String predType) {
		
	gRates.setTitle("Average "+predType+" Rate vs Batch Size");
	gRates.setTitleX("Batch Size");
	gRates.setTitleY(predType+" Rate [Hz]");
	gRates.setMarkerColor(3);
	gRates.setMarkerStyle(8);

	masterCanvas.getCanvas("EvRate").draw(gRates,"AP");
	masterCanvas.getCanvas("EvRate").getPad(0).setTitle("Average "+predType+" Rate vs Batch Size");
    }//End of PlotRatesVSBatchSize
	
    /*
     * Plots the Accuracy, Efficiency and Purity of the neural network as a function of the lower
     * threshold on the response. Returns the threshold at which the Purity is maximised whilst keeping
     * the Efficiency above 0.995.
     * 
     * Arguments:
     * 			NEvents: The length of the predictions array.
     * 			predictions: Contains the output of the neural network
     * 			Labels: Contains the true classes for each prediction
     * 
     * Returns:
     * 			Returns the threshold on the response at which the Purity is maximised whilst keeping
     * the Efficiency above 0.995.
     */
    public static double PlotMetricsVsResponse(int NEvents,INDArray predictions, INDArray Labels) {
	GraphErrors gAcc= new GraphErrors();
	GraphErrors gEff= new GraphErrors();
	GraphErrors gPur= new GraphErrors();
	double bestRespTh=0;
	double bestPuratEff0p995=0;
		
	//Loop over threshold on the response
	for(double RespTh=0.01; RespTh<0.99;RespTh+=0.01) {
	    INDArray metrics =getMetrics(NEvents, predictions, Labels, RespTh);
	    double Acc=metrics.getFloat(0,0);
	    double Pur=metrics.getFloat(1,0);
	    double Eff=metrics.getFloat(2,0);
	    gAcc.addPoint(RespTh, Acc, 0, 0);
	    gPur.addPoint(RespTh, Pur, 0, 0);
	    gEff.addPoint(RespTh, Eff, 0, 0);
	    if(Eff>0.995) {
		if (Pur>bestPuratEff0p995) {
		    bestPuratEff0p995=Pur;
		    bestRespTh=RespTh;
		}
	    }
	}//Increment threshold on response
		
	System.out.format("%n Best Purity at Efficiency above 0.995: %.3f at a threshold on the response of %.3f %n%n",bestPuratEff0p995,bestRespTh);
		
	gAcc.setTitle("Accuracy (Green)");
	gAcc.setTitleX("Classifier Response");
	gAcc.setTitleY("Metrics");
	gAcc.setMarkerColor(3);
	gAcc.setMarkerStyle(8);
	
		
	gPur.setTitle("Purity (Red)");
	gPur.setTitleX("Classifier Response");
	gPur.setTitleY("Metrics");
	gPur.setMarkerColor(2);
	gPur.setMarkerStyle(8);
	
		
	gEff.setTitle("Efficiency (Blue)");
	gEff.setTitleX("Classifier Response");
	gEff.setTitleY("Metrics");
	gEff.setMarkerColor(4);
	gEff.setMarkerStyle(8);

	masterCanvas.getCanvas("MetVsResp").getPad(0).setLegend(true);
	masterCanvas.getCanvas("MetVsResp").getPad(0).setLegendPosition(400, 300);

	masterCanvas.getCanvas("MetVsResp").draw(gAcc,"AP");
	masterCanvas.getCanvas("MetVsResp").draw(gPur,"sameAP");
	masterCanvas.getCanvas("MetVsResp").draw(gEff,"sameAP");
	masterCanvas.getCanvas("MetVsResp").getPad(0).setTitle("Metrics vs Response");
		
	return bestRespTh;
    }//End of PlotMetricsVSResponse
	
    /*
     * Plots the event rates, and the time it took to make prediction on NPreds (6 per Event).
     * 
     * Arguments:
     * 			NPreds: The number of predictions made
     * 			hRates: Histogram of the event rates.
     * 			hTimes: Histogram of the time taken to make NPreds predictions.
     *
     */
    public static void PlotPredRates(int NPreds, H1F hRates, H1F hTimes) {
	TCanvas canvasRates = new TCanvas("Event Rate",800,500);
	canvasRates.setDefaultCloseOperation(TCanvas.EXIT_ON_CLOSE);
	canvasRates.setLocationRelativeTo(null);
	canvasRates.setVisible(true);
		
	hRates.setTitle("Event Rate");
	hRates.setTitleX("Event Rate [Hz]");
	hRates.setTitleY("Counts");
	hRates.setLineWidth(2);
	hRates.setLineColor(4);
	hRates.setFillColor(4);
	canvasRates.draw(hRates);
	canvasRates.getCanvas().getPad(0).setTitle(canvasRates.getTitle());
		
	TCanvas canvasTimes = new TCanvas("Prediction Time for "+NPreds+" predictions (6 per Event)",800,500);
	canvasTimes.setDefaultCloseOperation(TCanvas.EXIT_ON_CLOSE);
	canvasTimes.setLocationRelativeTo(null);
	canvasTimes.setVisible(true);
		
	hTimes.setTitle("Prediction Time for "+NPreds+" predictions (6 per Event)");
	hTimes.setTitleX("Prediction Time [s]");
	hTimes.setTitleY("Counts");
	hTimes.setLineWidth(2);
	hTimes.setLineColor(4);
	hTimes.setFillColor(4);
	canvasTimes.draw(hTimes);
	canvasTimes.getCanvas().getPad(0).setTitle(canvasTimes.getTitle());
		
    }//End of PlotPredRates
	
    /*
     * Plots the classifier response given the output of the classifier and the true classes.
     * 
     * Arguments:
     * 			NEvents: The length of the output array.
     * 			output: Contains the the neural network predictions.
     * 			Labels: Contains the true classes for each prediction.
     */
    public static void PlotResponse(int NEvents,INDArray output, INDArray Labels) {
	H1F hRespPos = new H1F("Response_Positive_Sample", 100, 0, 1);
	H1F hRespNeg = new H1F("Response_Negative_Sample", 100, 0, 1);
	//Sort predictions into those made on the positive/or negative samples
	for(int i=0;i<NEvents;i+=1) {
	    if(Labels.getFloat(i,0)==1) {
		hRespPos.fill(output.getFloat(i,0));
	    } else if(Labels.getFloat(i,0)==0) {
		hRespNeg.fill(output.getFloat(i,0));
	    }
	}
	      
		
	hRespPos.setTitle("Positive Sample Response (Blue)");
	hRespPos.setTitleX("Classifier Response");
	hRespPos.setTitleY("Counts");
	hRespPos.setLineWidth(2);
	hRespPos.setLineColor(4);
	hRespPos.setFillColor(4);
		
	hRespNeg.setTitle("Negative Sample Response (Red)");
	hRespNeg.setTitleX("Classifier Response");
	hRespNeg.setTitleY("Counts");
	hRespNeg.setLineWidth(3);
	hRespNeg.setLineColor(2);

	masterCanvas.getCanvas("Resp").getPad(0).getAxisY().setLog(true);
	masterCanvas.getCanvas("Resp").getPad(0).setLegend(true);
	masterCanvas.getCanvas("Resp").getPad(0).setLegendPosition(400, 300);

	masterCanvas.getCanvas("Resp").draw(hRespPos);
	masterCanvas.getCanvas("Resp").draw(hRespNeg,"same");
	masterCanvas.getCanvas("Resp").getPad(0).setTitle("Classifier Response");
		
		
    }//End of PlotResponse
	
    /*
     *  Create DC images for a given sector.
     *  Also places requirements on the number of superlayers with at least one hit
     *  
     * Arguments:
     *  		dchits: Bank containing information from the drift chambers.
     *  		sector: sector for which to create the image
     *  		minSL: minimal amount of superlayers hit (>=minSL)
     *  		maxSL: maximal amount of superlayers hit (<=maxSL)
     *  
     * Returns:
     * 			EC image for a given sector.
     */
    public static INDArray FillDCArray(Bank dchits, int sector, int minSL, int maxSL) {
	//Initialise array to all zeros
	INDArray DCVals = Nd4j.zeros(6,112);
	//Use following vector to check how many superlayers have at least one hit
	Vector<Integer> SLs = new Vector<Integer>();
	for (int k = 0; k < dchits.getRows(); k++) {
	    int sectorDC = dchits.getInt("sector", k);
	    if (sectorDC == sector) { //check that the hits are in the right sector
		int wire = dchits.getInt("wire", k);
		int superlayer = dchits.getInt("superlayer", k);
		//need to increment by 1/6 not assign 1/6!!
		double tempElement=DCVals.getDouble(superlayer-1,wire-1) + 1.0/6.0;
		//array index 0-5 not 1-6
		DCVals.putScalar(new int[] {superlayer-1,wire-1}, tempElement);
		if (!SLs.contains(superlayer)) {
		    SLs.add(superlayer);
		}
	    }
	}
	//check that we have enough superlayers hit
	if(SLs.size()>=minSL && SLs.size()<=maxSL) {
	    return DCVals;
	} else {
	    return Nd4j.zeros(6,112);
	}
    }//END FillDCArray

    /*
     *  Create DC background images for a given sector. Uses a different bank 
     *  structure than for tracks.
     *  
     * Arguments:
     *  		dchits: Bank containing information from the drift chambers.
     *  		sector: sector for which to create the image
     *  
     * Returns:
     * 			DC image for a given sector.
     */
    public static INDArray FillDCBgArray(Bank dchits, int sector) {
	//Initialise array to all zeros
	INDArray DCVals = Nd4j.zeros(6,112);
	for (int k = 0; k < dchits.getRows(); k++) {
	    int sectorDC = dchits.getInt("sector", k);
	    if (sectorDC == sector) { //check that the hits are in the right sector
		int layer = dchits.getInt("layer", k);
		int wire =  dchits.getInt("component", k);


		//Need to convert layers going from 1 to 36
		// (or 0 to 35 by taking away 1)
		// into sl going from 1 to 6
		// layer=(superlayer-1)*6 + n, n[0-5]
		// eg: layer=36 is in superlayer 6
		// eg: layer=15 is in superlayer 3
		int superlayer = (layer-1)/6 + 1;
		

		//System.out.println("layer "+layer+" superlayer: "+superlayer+" wire: "+wire);

		//need to increment by 1/6 not assign 1/6!!
		double tempElement=DCVals.getDouble(superlayer-1,wire-1) + 1.0/6.0;
	   
		//array index 0-5 not 1-6
		DCVals.putScalar(new int[] {superlayer-1,wire-1}, tempElement);
	    }
	}
	return DCVals;
    }//END FillDCBgArray
	

    /*
     *  Create EC images for a given sector.
     *  
     * Arguments:
     *  		echits: Bank containing information from the FD calorimeters.
     *  		sector: sector for which to create the image
     *  
     * Returns:
     * 			EC image for a given sector.
     */
    public static INDArray FillECArray(Bank echits, int sector) {
	//Initialise array to all zeros
	INDArray ECVals = Nd4j.zeros(6,72);
		
	for (int k = 0; k < echits.getRows(); k++) {
	    float energy = echits.getFloat("energy", k)/3;
	    int strip = echits.getInt("strip", k);
	    int sectorEC = echits.getInt("sector", k);
	    int layer=echits.getInt("layer", k);
	    if(sectorEC==sector) {//check that the hits are in the right sector
		//Layer 1-3: PCAL, 4-6: ECin, 7-9: ECout
		//Array indexing Rows 0-2: PCAL, 3-5: ECin + ECout (strips 0-71)
		//Array indexing columns: 0-35: ECin, 36-71: ECout
		if(layer>6) {
		    strip=strip+36;
		    layer=layer-3;
		} 
		ECVals.putScalar(new int[] {layer-1,strip-1}, energy);
				
	    }
	}//loop over echits rows
	return ECVals;
    }//END FillECArray
	
	
    /*
     * Returns the data parsed into the correct format for the AI Trigger classifier.
     * This is meant for testing purposes, and therefore returns a ND4J MultiDataset
     * that also contains the true Labels for each event and the momentum of the DC track.
     * 
     * Arguments:
     *			nPreds: The number of predictions desired.
     *			minSL: The minimum amount of superlayers with a hit
     *			maxSL: The maximal amount of superlayers with a hit
     * 
     * Returns:
     * 			ND4J MultiDataSet containing DC and EC images for both positive and negative sample events,
     * the Labels used to distinguish these and the momentum of the particle associated with the DC track.
     * 	 
     */
    public static MultiDataSet ParseDataForTesting(String fName2,int nPreds, int minSL, int maxSL) {
	INDArray DCArray=Nd4j.zeros(nPreds,6,112,1);
	INDArray ECArray=Nd4j.zeros(nPreds,6,72,1);
	INDArray Labels=Nd4j.zeros(nPreds,2);
	INDArray Ps=Nd4j.zeros(nPreds,1);
	HipoReader reader = new HipoReader();
	reader.open(fName2);
	Event event = new Event();
	Bank dchits = new Bank(reader.getSchemaFactory().getSchema("TimeBasedTrkg::TBHits"));
	Bank echits = new Bank(reader.getSchemaFactory().getSchema("ECAL::hits"));
	Bank parts = new Bank(reader.getSchemaFactory().getSchema("REC::Particle"));
	Bank track = new Bank(reader.getSchemaFactory().getSchema("REC::Track"));
	int nPred=0, nPosPred=0, nNegPred=0;
	while (reader.hasNext() == true && nPred<nPreds) {
	    reader.nextEvent(event);
	    event.read(dchits);
	    event.read(echits);
	    event.read(parts);
	    event.read(track);
			
	    /* Use the following vectors to store the pindexes of electrons in the event,
	     * and the sectors were they were reconstructed.
	     * pByIndex and pBySector allow to store the momentum of the particle
	     * at a given pindex or in a given sector.
	     */
	    Vector<Integer> elIndexes = new Vector<Integer>();
	    Vector<Integer> elSectors = new Vector<Integer>();
	    Map<Integer, Double> pByIndex= new HashMap<Integer,Double>();
	    Map<Integer, Double> pBySector= new HashMap<Integer,Double>();
	    for (int i = 0; i < parts.getRows(); i++) {
		int pid = parts.getInt("pid", i);
		int status = parts.getInt("status", i);
		double px=parts.getFloat("px", i);
		double py=parts.getFloat("py", i);
		double pz=parts.getFloat("pz", i);
		if (Math.abs(status) >= 2000 && Math.abs(status) < 4000) {
		    pByIndex.put(i,Math.sqrt(px*px+py*py+pz*pz));
		    if (pid == 11) {
			elIndexes.add(i);
		    }
		}
	    }
	    /*
	     * Unfortunately the sector of a particle isn't contained in the REC::Particle bank
	     * so we use the track bank, which has pindex and sector information.
	     */
	    for (int k = 0; k < track.getRows(); k++) {
		int pindex = track.getInt("pindex", k);
		int sectorTrk = track.getInt("sector", k);
		if(pByIndex.get(pindex)!=null) {
		    pBySector.put(sectorTrk,pByIndex.get(pindex));
		}
		if (elIndexes.contains(pindex)) {
		    elSectors.add(sectorTrk);
		}
	    }
			
	    //loop over each sector
	    for(int sector=1;sector<7;sector++) {
		INDArray EventDCArray=FillDCArray(dchits,sector, minSL, maxSL);
		INDArray EventECArray=FillECArray(echits,sector);
				
		if(EventDCArray.any() && EventECArray.any()) { //check that the images aren't all empty
		    if(elSectors.contains(sector) && nPosPred<nPreds) { //check for reconstructed electron in sector
			if(nPosPred<(nPreds/2)){
			    DCArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(EventDCArray);
			    ECArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(EventECArray);
			    Labels.putScalar(new int[] {nPred,0}, 1); //column 0 set to 1 (both cols initialised to 0)
			    if(pBySector.get(sector)!=null) {
				Ps.putScalar(new int[] {nPred,0}, pBySector.get(sector));
			    }
			    nPosPred++;
			    nPred++;
			}
		    } else {
			if(nNegPred<nPosPred) {//Make sure to balance dataset
			    DCArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(EventDCArray);
			    ECArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(EventECArray);
			    Labels.putScalar(new int[] {nPred,1}, 1);//column 1 set to 1 (both cols initialised to 0)
			    if(pBySector.get(sector)!=null) {
				Ps.putScalar(new int[] {nPred,0}, pBySector.get(sector));
			    }
			    nNegPred++;
			    nPred++;
			}
		    }//Electron or not 
		}// Only take non null arrays
	    }// Loop over sectors
	} //Read Event
	System.out.println("Number of Predictions "+nPred+" Positive: "+nPosPred+" Negative: "+nNegPred);
	INDArray[] toDS=new INDArray[2];
	toDS[0]=DCArray;
	toDS[1]=ECArray;
	INDArray[] toDSLabels=new INDArray[2];
	toDSLabels[0]=Labels;
	toDSLabels[1]=Ps;
	MultiDataSet dataset = new MultiDataSet(toDS,toDSLabels);
	return dataset;
    }//End of ParseDataForTesting

 /*
     * Returns the background data parsed into the correct format for the AI Trigger classifier.
     * 
     * 
     * Arguments:
     *			nPreds: The number of predictions desired.
     * 
     * Returns:
     * 			ND4J MultiDataSet containing DC images for background events
     * 
     * 	 
     */
    public static MultiDataSet ParseBackground(String fName2,int nPreds) {
	INDArray DCArray=Nd4j.zeros(nPreds,6,112,1);
	INDArray Labels=Nd4j.zeros(nPreds,2);
	INDArray Ps=Nd4j.zeros(nPreds,1);
	HipoReader reader = new HipoReader();
	reader.open(fName2);
	Event event = new Event();
	Bank dchits = new Bank(reader.getSchemaFactory().getSchema("DC::tdc"));
	int nPred=0;
	while (reader.hasNext() == true) {
	    reader.nextEvent(event);
	    event.read(dchits);
			
	    //loop over each sector
	    for(int sector=1;sector<7;sector++) {
		INDArray EventDCArray=FillDCBgArray(dchits,sector);				
		if(EventDCArray.any() && nPred<nPreds) { //check that the images aren't all empty
		    DCArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(EventDCArray);
		    nPred++;
		}// Only take non null arrays
	    }// Loop over sectors
	} //Read Event
	System.out.println("Number of Predictions "+nPred);
	INDArray[] toDS=new INDArray[2];
	toDS[0]=DCArray;
	INDArray[] toDSLabels=new INDArray[2];
	toDSLabels[0]=Labels;
	toDSLabels[1]=Ps;
	MultiDataSet dataset = new MultiDataSet(toDS,toDSLabels);
	return dataset;
    }//End of ParseBackground
	
	
    /*
     * Returns INDArray list with the first entry being a NEvents*2 array, the first column is 1 (0) 
     * if the C12Trigger has been called (or not) the second column is 1 (0) if an electron is
     * reconstructed in the same sector (or not). The second entry is the momentum of the track
     *  in which the sector has been called.
     */
    public static INDArray[] GetFileTriggerAndTruth(int NEvents) {
	INDArray Trigger=Nd4j.ones(NEvents,2);
	INDArray P=Nd4j.zeros(6*NEvents,1);
	HipoReader reader = new HipoReader();
	reader.open(fName);
	Event event = new Event();
	Bank parts = new Bank(reader.getSchemaFactory().getSchema("REC::Particle"));
	Bank track = new Bank(reader.getSchemaFactory().getSchema("REC::Track"));
	Bank config = new Bank(reader.getSchemaFactory().getSchema("RUN::config"));
	int nPred=0;
		
	//loop over all events in file until reach limit
	while (reader.hasNext() == true && nPred<NEvents) {
	    reader.nextEvent(event);
	    event.read(parts);
	    event.read(track);
	    event.read(config);

	    /* Use the following vectors to store the pindexes of electrons in the event,
	     * and the sectors were they were reconstructed.
	     * pByIndex and pBySector allow to store the momentum of the particle
	     * at a given pindex or in a given sector.
	     */
	    Vector<Integer> elIndexes = new Vector<Integer>();
	    Vector<Integer> elSectors = new Vector<Integer>();
	    Map<Integer, Double> pByIndex= new HashMap<Integer,Double>();
	    Map<Integer, Double> pBySector= new HashMap<Integer,Double>();
	    for (int i = 0; i < parts.getRows(); i++) {
		int pid = parts.getInt("pid", i);
		int status = parts.getInt("status", i);
		double px=parts.getFloat("px", i);
		double py=parts.getFloat("py", i);
		double pz=parts.getFloat("pz", i);
		if (Math.abs(status) >= 2000 && Math.abs(status) < 4000) {
		    pByIndex.put(i,Math.sqrt(px*px+py*py+pz*pz));
		    if (pid == 11) {
			elIndexes.add(i);
		    }
		}
	    }//loop over particle bank
			
	    /*
	     * Unfortunately the sector of a particle isn't contained in the REC::Parts bank
	     * so we use the track bank, which has pindex and sector information.
	     */
	    for (int k = 0; k < track.getRows(); k++) {
		int pindex = track.getInt("pindex", k);
		int sectorTrk = track.getInt("sector", k);
		if(pByIndex.get(pindex)!=null) {
		    pBySector.put(sectorTrk,pByIndex.get(pindex));
		}
		if (elIndexes.contains(pindex)) {
		    elSectors.add(sectorTrk);
		}
	    }//loop over track banks to get sector
			
	    //Loop over config bank to check if the trigger was set in a sector
	    for(int k = 0; k < config.getRows(); k++) {
		for(int sector=1; sector<7;sector++) {
		    if((config.getLong("trigger", k) & (1<<sector))!=0) { //check if trigger bit is set in sector
			if(elSectors.contains(sector)) { // check if there's a reconstructed electron in the sector
			    if(pBySector.get(sector)!=null) {
				P.putScalar(new int[] {nPred,0}, pBySector.get(sector));
			    }
			    nPred++;
			} else {
			    Trigger.putScalar(new int[] {nPred,1}, 0);
			    if(pBySector.get(sector)!=null) {
				P.putScalar(new int[] {nPred,0}, pBySector.get(sector));
			    }
			    nPred++;
			}
		    }
		}
			
	    }
			
	}// While Reading events
	//Creates list of INDArray outputs
	INDArray[] output=new INDArray[2];
	output[0]=Trigger;
	output[1]=P;
	return output;
    }//End of GetFileTriggerAndTrut

}
