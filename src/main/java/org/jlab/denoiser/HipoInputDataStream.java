/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.denoiser;

import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @authors gavalian, tyson
 */
public class HipoInputDataStream implements InputDataStream {
    HipoReader reader = null;
    Event event = null;
    Bank dchits = null;

    HipoReader readerNoise = null;
    Event eventNoise = null;
    Bank dchitsNoise = null;
    
    int batchSize=100;//Define as number of events with six sectors per event?
    
    /**
	 *  Opens a .hipo file at the specified location
	 *  
	 * Arguments:
	 *  		url: location of the hipo file
	 *  
	 */
    public void open(String url,String urlNoise) {
    	reader = new HipoReader();
    	reader.open(url);
    	event = new Event();
    	dchits = new Bank(reader.getSchemaFactory().getSchema("TimeBasedTrkg::TBHits"));

	readerNoise = new HipoReader();
    	readerNoise.open(urlNoise);
    	eventNoise = new Event();
    	dchitsNoise = new Bank(readerNoise.getSchemaFactory().getSchema("DC::tdc"));
    }
    /**
	 *  Sets the batch size defined in number of events, with 6 entries per event
	 *  ie one for each sector
	 *  
	 * Arguments:
	 *  		size: the desired batch size.
	 */
    public void setBatch(int size) {batchSize=size;}
    
    /**
	 *  Reads the hipo file and creates arrays from the relevant DC and ECAL banks.
	 *  
	 * Returns:
	 * 			INDArray list with the DC info at index 0 and the ECAL info at index 1.
	 */
    public INDArray[] next() {
    	INDArray DCArray=Nd4j.zeros(6*batchSize,6,112,1);
	INDArray DCArrayNoise=Nd4j.zeros(6*batchSize,6,112,1);
    	int nPred=0;
    	//read all events until limit
    	while (reader.hasNext() == true && readerNoise.hasNext() == true && nPred<(batchSize*6)) {
        	reader.nextEvent(event);
        	event.read(dchits);

		readerNoise.nextEvent(eventNoise);
        	eventNoise.read(dchitsNoise);
        	//Fill output arrays for each sector
        	for(int sector=1;sector<7;sector++) {
            	DCArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(FillDCArray(sector));	
		DCArrayNoise.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(FillDCNoiseArray(sector));	
            	nPred++;
            }
        }
    	//Creates list for output
    	INDArray[] output=new INDArray[3];
    	output[0]=DCArray;
    	output[1]=DCArrayNoise;
	output[2]=DCArray.add(DCArrayNoise);
    	return output;
    	
    }//END next
    
    /**
	 *  Create DC images for a given sector from the dchits bank.
	 *  Also places requirements on the number of superlayers with at least one hit
	 *  
	 * Arguments:
	 *  		sector: sector for which to create the image
	 *  
	 * Returns:
	 * 			DC image for a given sector.
	 */
	private INDArray FillDCArray(int sector) {
    	//Initialise array to all zeros
    	INDArray DCVals = Nd4j.zeros(6,112);
    	for (int k = 0; k < dchits.getRows(); k++) {
        	int sectorDC = dchits.getInt("sector", k);
        	if (sectorDC == sector) { //check that the hits are in the right sector
            	int wire = dchits.getInt("wire", k);
            	int superlayer = dchits.getInt("superlayer", k);
            	//need to increment by 1/6 not assign 1/6!!
            	double tempElement=DCVals.getDouble(superlayer-1,wire-1) + 1.0/6.0;
            	//array index 0-5 not 1-6
            	DCVals.putScalar(new int[] {superlayer-1,wire-1}, tempElement);
            }
        }
    	return DCVals;
	}//END FillDCArray
	

	 /*
     *  Create DC noise images for a given sector. Uses a different bank 
     *  structure than for tracks.
     *  
     * Arguments:
     *  		dchits: Bank containing information from the drift chambers.
     *  		sector: sector for which to create the image
     *  
     * Returns:
     * 			DC image for a given sector.
     */
    private INDArray FillDCNoiseArray(int sector) {
	//Initialise array to all zeros
	INDArray DCVals = Nd4j.zeros(6,112);
	for (int k = 0; k < dchitsNoise.getRows(); k++) {
	    int sectorDC = dchitsNoise.getInt("sector", k);
	    if (sectorDC == sector) { //check that the hits are in the right sector
		int layer = dchitsNoise.getInt("layer", k);
		int wire =  dchitsNoise.getInt("component", k);
		//Need to convert layers going from 1 to 36
		// (or 0 to 35 by taking away 1)
		// into sl going from 1 to 6
		// layer=(superlayer-1)*6 + n, n[0-5]
		// eg: layer=36 is in superlayer 6
		// eg: layer=15 is in superlayer 3
		int superlayer = (layer-1)/6 + 1;
	        
		//need to increment by 1/6 not assign 1/6!!
		double tempElement=DCVals.getDouble(superlayer-1,wire-1) + 1.0/6.0;
		//array index 0-5 not 1-6
		DCVals.putScalar(new int[] {superlayer-1,wire-1}, tempElement);
	    }
	}
	return DCVals;
    }//END FillDCBgArray
    
	 /**
	  *  Check if there's more events to be read.
	  *  
	  * Returns:
	  * 			True if the hipo reader still has more events to read.
	  */
    public boolean hasNext() {return reader.hasNext();}

    /**
	  *  Check if there's more noise events to be read.
	  *  
	  * Returns:
	  * 			True if the hipo reader still has more events to read.
	  */
    public boolean hasNextNoise() {return readerNoise.hasNext();}
    
    /**
	  *  Do something with the trigger model predictions
	  *  
	  * Arguments:
	  * 			resutls: INDArray containing the model predictions.
	  */
    public void apply(INDArray[] result) {
    	
    }//END apply
    
}//EndOfClass
