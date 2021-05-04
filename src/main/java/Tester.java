import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import javax.swing.JFrame;

import org.jlab.groot.data.GraphErrors;
import org.jlab.groot.data.H1F;
import org.jlab.groot.graphics.EmbeddedCanvas;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;



/*	GPU Stuff:
 * check https://deeplearning4j.konduit.ai/config/backends/config-cudnn
 *       https://deeplearning4j.konduit.ai/config/backends
 *       
 * Redist for version 10.2
  <dependency>
    <groupId>org.bytedeco</groupId>
	    <artifactId>cuda-platform-redist</artifactId>
    <version>10.2-7.6-1.5.3</version>
  </dependency>
 * 
 * 
 * artifact ID:
 * For deeplearning4j:
 * deeplearning4j-core for CPU
 * deeplearning4j-cuda-10.0 (or 10.1,10.2)
 * 
 * For nd4j:
 * nd4j-native-platform when using CPU(or just nd4j-native)
 * nd4j-cuda-10.2 when using GPU. Available CUDA version are 9.2, 10, 10.1, 10.2
 */



public class Tester {
	//Need to change the following to point to the input .hipo file
	static String fName="/w/work5/jlab/hallb/clas12/rg-a/trackingInfo/out_clas_005038.evio.00105-00109.hipo";
	
	/*
	 * Main Function, runs the different relevant tests.
	 */
	public static void main(String[] args) {
		
		/*
		 * When we initialise the Clas12AIElTriggerHipo we need to pass the number of events to predict on.
		 * Each event has 6 predictions, one per sector.
		 * We can then read and predict on these events using the Predict function.
		 */
		int NEvents=100;
		Clas12AIElTriggerHipo AITrigger = new Clas12AIElTriggerHipo(fName,NEvents);
		int BatchSize=600;
		
		Instant start = Instant.now();
		INDArray output=AITrigger.Predict(BatchSize);
		Instant end = Instant.now();
		double timeElapsed = Duration.between(start, end).toNanos()*1e-9; 
		double rate=BatchSize/timeElapsed;
		System.out.format("%nTook %.3f to read and predict on %d Events for Batch Size of %d %n",timeElapsed,NEvents,BatchSize);
		System.out.format(" ie rate of %.3f events per second %n%n",rate);
		
		/*
		 * Here we want to reload data for testing. This is due to the fact that for testing we need to sort
		 * samples based on whether there was an electron or not in the sector.
		 * Instead of reading in 6*NEvents we can just pass the number of predictions directly.
		 * The ParseDataForTesting returns our testing dataset, which we'll then use for different tests.
		 */
		MultiDataSet Data=AITrigger.ParseDataForTesting(NEvents*10*6,5,6);
		String networkLoc=AITrigger.GetNetworkLocation();
		
		/*
		 * Here we measure the EventRate as a function of BatchSize.
		 */
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
			if(avRate>bestAvRate) {
				bestAvRate=avRate;
				bestBatchSize=BatchSizeT;
			}
		}
		System.out.format("Best Event Rate %.3f for Batch Size of %d %n%n",bestAvRate,bestBatchSize);
		PlotRatesVSBatchSize(gRates, "Prediction");
		PlotRatesVSBatchSize(gRatesEvent, "Event");
		
		/*
		 * Here we plot the metrics used to evaluate the AI Trigger by specifying verbose=true.
		 */
		
		int NBatches=(NEvents*10*6)/BatchSize;
		boolean verbose=true;
		Test(Data, NBatches, BatchSize,1,verbose,networkLoc);
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
			System.out.println("Re-reading File to Calculate C12 Trigger Purity:");
			Clas12AIElTriggerHipo dp = new Clas12AIElTriggerHipo(fName, NEvents);
			C12TriggerAndP= dp.GetFileTriggerAndTruth();
			System.out.format("%n C12Trigger Purity: %.3f%n%n",calculateCLAS12TriggerPurity(NEvents, C12TriggerAndP[0]));
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
		
		JFrame frameMP = new JFrame("Metrics vs Momentum");
		frameMP.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		EmbeddedCanvas canvasMP = new EmbeddedCanvas();
		frameMP.setSize(800, 500);
		canvasMP.getPad(0).setTitle("Metrics vs Momentum");
		
		canvasMP.draw(gEff,"AP");
		canvasMP.draw(gPur,"sameAP");
		
		canvasMP.getPad(0).setLegend(true);
		canvasMP.getPad(0).setLegendPosition(400, 300);
		frameMP.add(canvasMP);
		frameMP.setLocationRelativeTo(null);
		frameMP.setVisible(true);
		
		if(wC12TriggerComp) {
			gC12Pur.setTitle("CLAS12 Trigger Purity (Green)");
			gC12Pur.setTitleX("Momentum [GeV]");
			gC12Pur.setTitleY("Metrics");
			gC12Pur.setMarkerColor(3);
			gC12Pur.setMarkerStyle(8);
			
			
			JFrame frameMPwC12 = new JFrame("Metrics vs Momentum with C12 Trigger");
			frameMPwC12.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			EmbeddedCanvas canvasMPwC12 = new EmbeddedCanvas();
			frameMPwC12.setSize(800, 500);
			canvasMPwC12.getPad(0).setTitle("Metrics vs Momentum");
			
			canvasMPwC12.draw(gEff,"AP");
			canvasMPwC12.draw(gPur,"sameAP");
			canvasMPwC12.draw(gC12Pur,"sameAP");
			
			canvasMPwC12.getPad(0).setLegend(true);
			canvasMPwC12.getPad(0).setLegendPosition(400, 150);
			frameMPwC12.add(canvasMPwC12);
			frameMPwC12.setLocationRelativeTo(null);
			frameMPwC12.setVisible(true);
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
		
		JFrame frameR = new JFrame("Average "+predType+" Rate vs Batch Size");
		frameR.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		EmbeddedCanvas canvasR = new EmbeddedCanvas();
		frameR.setSize(800, 500);
		canvasR.getPad(0).setTitle("Average "+predType+" Rate vs Batch Size");
		
		gRates.setTitle("Average "+predType+" Rate vs Batch Size");
		gRates.setTitleX("Batch Size");
		gRates.setTitleY(predType+" Rate [Hz]");
		gRates.setMarkerColor(3);
		gRates.setMarkerStyle(8);
		canvasR.draw(gRates,"AP");
		
		frameR.add(canvasR);
		frameR.setLocationRelativeTo(null);
		frameR.setVisible(true);
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
		
		JFrame frameMR = new JFrame("Metrics vs Threshold on Response");
		frameMR.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		EmbeddedCanvas canvasMR = new EmbeddedCanvas();
		frameMR.setSize(800, 500);
		canvasMR.getPad(0).setTitle("Metrics vs Threshold on Response");
		
		gAcc.setTitle("Accuracy (Green)");
		gAcc.setTitleX("Classifier Response");
		gAcc.setTitleY("Metrics");
		gAcc.setMarkerColor(3);
		gAcc.setMarkerStyle(8);
		canvasMR.draw(gAcc,"AP");
		
		gPur.setTitle("Purity (Red)");
		gPur.setTitleX("Classifier Response");
		gPur.setTitleY("Metrics");
		gPur.setMarkerColor(2);
		gPur.setMarkerStyle(8);
		canvasMR.draw(gPur,"sameAP");
		
		gEff.setTitle("Efficiency (Blue)");
		gEff.setTitleX("Classifier Response");
		gEff.setTitleY("Metrics");
		gEff.setMarkerColor(4);
		gEff.setMarkerStyle(8);
		canvasMR.draw(gEff,"sameAP");
		
		canvasMR.getPad(0).setLegend(true);
		canvasMR.getPad(0).setLegendPosition(400, 300);
		frameMR.add(canvasMR);
		frameMR.setLocationRelativeTo(null);
		frameMR.setVisible(true);
		
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
		JFrame frameRates = new JFrame("Event Rate");
		frameRates.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		EmbeddedCanvas canvasRates = new EmbeddedCanvas();
		canvasRates.getPad(0).getAxisY().setLog(true);
		frameRates.setSize(800, 500);
		canvasRates.getPad(0).setTitle("Event Rate");
		
		hRates.setTitle("Event Rate");
		hRates.setTitleX("Event Rate [Hz]");
		hRates.setTitleY("Counts");
		hRates.setLineWidth(2);
		hRates.setLineColor(4);
		hRates.setFillColor(4);
		canvasRates.draw(hRates);
		
		frameRates.add(canvasRates);
		frameRates.setLocationRelativeTo(null);
		frameRates.setVisible(true);
		
		JFrame frameTimes = new JFrame("Prediction Time for "+NPreds+" predictions (6 per Event)");
		frameTimes.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		EmbeddedCanvas canvasTimes = new EmbeddedCanvas();
		canvasTimes.getPad(0).getAxisY().setLog(true);
		frameTimes.setSize(800, 500);
		canvasTimes.getPad(0).setTitle("Prediction Time for "+NPreds+" predictions (6 per Event)");
		
		hTimes.setTitle("Prediction Time for "+NPreds+" predictions (6 per Event)");
		hTimes.setTitleX("Prediction Time [s]");
		hTimes.setTitleY("Counts");
		hTimes.setLineWidth(2);
		hTimes.setLineColor(4);
		hTimes.setFillColor(4);
		canvasTimes.draw(hTimes);
		
		frameTimes.add(canvasTimes);
		frameTimes.setLocationRelativeTo(null);
		frameTimes.setVisible(true);
		
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
		
		JFrame frameResp = new JFrame("Classifier Response");
		frameResp.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		EmbeddedCanvas canvasResp = new EmbeddedCanvas();
		canvasResp.getPad(0).getAxisY().setLog(true);
		frameResp.setSize(800, 500);
		canvasResp.getPad(0).setTitle("Classifier Response");
		
		hRespPos.setTitle("Positive Sample Response");
		hRespPos.setTitleX("Classifier Response");
		hRespPos.setTitleY("Counts");
		hRespPos.setLineWidth(2);
		hRespPos.setLineColor(4);
		hRespPos.setFillColor(4);
		canvasResp.draw(hRespPos);
		
		hRespNeg.setTitle("Negative Sample Response");
		hRespNeg.setTitleX("Classifier Response");
		hRespNeg.setTitleY("Counts");
		hRespNeg.setLineWidth(3);
		hRespNeg.setLineColor(2);
		canvasResp.draw(hRespNeg,"same");
		
		canvasResp.getPad(0).setLegend(true);
		canvasResp.getPad(0).setLegendPosition(100, 20);
		frameResp.add(canvasResp);
		frameResp.setLocationRelativeTo(null);
		frameResp.setVisible(true);
		
	}//End of PlotResponse

}
