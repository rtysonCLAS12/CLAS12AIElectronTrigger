import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


public class Clas12AIElTriggerHipo implements Clas12AIElTrigger {
	String fName; 
	int NEvents;
	String networkLoc="trained_model.h5";
	ComputationGraph network;
	
	/*
	 * Constructor, initialises
	 */
	public Clas12AIElTriggerHipo(String fNameIn, int NEventsIn) {
		fName=fNameIn;
		NEvents=NEventsIn;
		LoadNetwork();
	}
	
	/*
	 * Load AI Trigger neural network from hardcoded location.
	 * Will give an exception if thenetwork cannot be loaded. 
	 */
	public void LoadNetwork() {
		try {
			network = KerasModelImport.importKerasModelAndWeights(networkLoc);
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
	}//End of LoadNetwork
	
	/*
	 * Returns the predictions made by the AI Trigger classifier on data loaded and prepared by parseData.
	 * Predictions are made per batches.
	 * Will give an exception if the network isn't loaded correctly.
	 * 
	 * Arguments:
	 * 			BatchSize: Size of batch of predictions, number of predictions must be divisible by BatchSize!!
	 * 
	 * Returns:
	 * 			The response of the classifier as an NPrediction*2 INDArray with the probability that
	 * an electron is in that sector in the first column, and the probability that there isn't an electron
	 * in that sector in the second column.
	 * 	 
	 */
	public INDArray Predict(int BatchSize) {
		//Parse Data into correct format for AI Trigger
		INDArray[] Data=ParseData();

		long NEvents=Data[0].shape()[0];
		long NBatches=NEvents/BatchSize;
		INDArray networkOutput=Nd4j.zeros(NEvents,2);

		networkOutput=network.output(Data)[0];
		//Make predictions per batch
		for(int batch=0; batch<NBatches;batch++) {
			int batchStart=batch*BatchSize;
			int batchEnd=(batch+1)*BatchSize;
				
			//Separate DC and EC data into batches
			INDArray[] DataBatch=new INDArray[2];
			DataBatch[0]=Data[0].get(NDArrayIndex.interval(batchStart,batchEnd), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
			DataBatch[1]=Data[1].get(NDArrayIndex.interval(batchStart,batchEnd), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
				
			//networkOutput contains the predictions for all batches
			networkOutput.get(NDArrayIndex.interval(batchStart,batchEnd), NDArrayIndex.all()).assign(network.output(DataBatch)[0]);
		}	
		return networkOutput;
	}//End Of Predict
	
	
	/*
	 * Returns the data parsed into the correct format for the AI Trigger classifier.
	 *
	 * 
	 * Returns:
	 * 			INDArray List containing the DC images in Data[0], and EC images in Data[1]
	 * 	 
	 */
	public INDArray[] ParseData() {
		//Initialise arrays to 0
		INDArray DCArray=Nd4j.zeros(6*NEvents,6,112,1);
		INDArray ECArray=Nd4j.zeros(6*NEvents,6,72,1);
		
		//Open and set up HipoReader
		HipoReader reader = new HipoReader();
		reader.open(fName);
		Event event = new Event();
		
		//Initialise relevant banks
		Bank dchits = new Bank(reader.getSchemaFactory().getSchema("TimeBasedTrkg::TBHits")); //NB: need to change to dc::hits (?)
		Bank echits = new Bank(reader.getSchemaFactory().getSchema("ECAL::hits"));
		int nPred=0;
		
		//read all events until limit
		while (reader.hasNext() == true && nPred<(NEvents*6)) {
			reader.nextEvent(event);
			event.read(dchits);
			event.read(echits);
			//Fill output arrays for each sector
			for(int sector=1;sector<7;sector++) {
				DCArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(FillDCArray(dchits,sector,0,7));
				ECArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(FillECArray(echits,sector));		
				nPred++;
			}
		}
		//Creates list for output
		INDArray[] output=new INDArray[2];
		output[0]=DCArray;
		output[1]=ECArray;
		return output;
	}//END ReadFile
	
	
	
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
	public INDArray FillDCArray(Bank dchits, int sector, int minSL, int maxSL) {
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
	 *  Create EC images for a given sector.
	 *  
	 * Arguments:
	 *  		echits: Bank containing information from the FD calorimeters.
	 *  		sector: sector for which to create the image
	 *  
	 * Returns:
	 * 			EC image for a given sector.
	 */
	public INDArray FillECArray(Bank echits, int sector) {
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
	 * Applies a threshold to the classifier output (response). This varies from 0 to 1, and so we
	 * round this to 1 (0) if the response is above (equal or below) the threshold.
	 * 
	 * Arguments:
	 * 			Predictions: An INDArray containing the classifier output. The is the probability that
	 *  an event is of the positive sample in column 0, and of the negative sample in column 1. 
	 * 			Threshold: The desired threshold on the response.
	 * 
	 * Returns:
	 * 			The response of the classifier rounded based on the inputed threshold.
	 * 	 
	 */
	public int[] ApplyResponseThreshold(INDArray Predictions, double Threshold){
		int NPreds=(int) Predictions.shape()[0];
		int[] roundedPredictions=new int[NPreds];
		for(int i=0;i<NPreds;i+=1) {
			if(Predictions.getFloat(i,0)>Threshold) {
				roundedPredictions[i]=1;
			} else {
				roundedPredictions[i]=0;
			}
		}
		return roundedPredictions;
	}
	
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
	public MultiDataSet ParseDataForTesting(int nPreds, int minSL, int maxSL) {
		INDArray DCArray=Nd4j.zeros(nPreds,6,112,1);
		INDArray ECArray=Nd4j.zeros(nPreds,6,72,1);
		INDArray Labels=Nd4j.zeros(nPreds,2);
		INDArray Ps=Nd4j.zeros(nPreds,1);
		HipoReader reader = new HipoReader();
		reader.open(fName);
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
						DCArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(EventDCArray);
						ECArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(EventECArray);
						Labels.putScalar(new int[] {nPred,0}, 1); //column 0 set to 1 (both cols initialised to 0)
						if(pBySector.get(sector)!=null) {
							Ps.putScalar(new int[] {nPred,0}, pBySector.get(sector));
						}
						nPosPred++;
						nPred++;
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
	 * Returns INDArray list with the first entry being a NEvents*2 array, the first column is 1 (0) 
	 * if the C12Trigger has been called (or not) the second column is 1 (0) if an electron is
	 * reconstructed in the same sector (or not). The second entry is the momentum of the track
	 *  in which the sector has been called.
	 */
	public INDArray[] GetFileTriggerAndTruth() {
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
	
	/*
	 * Returns the neural network location.
	 */
	public String GetNetworkLocation() {
		return networkLoc;
	}
	
}//END CLASS
