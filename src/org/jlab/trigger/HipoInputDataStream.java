/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.trigger;

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
    Bank echits = null;
    
    int batchSize=100;//Define as number of events with six sectors per event?
    
    /**
	 *  Opens a .hipo file at the specified location
	 *  
	 * Arguments:
	 *  		url: location of the hipo file
	 *  
	 */
    public void open(String url) {
    	reader = new HipoReader();
    	reader.open(url);
    	event = new Event();
    	dchits = new Bank(reader.getSchemaFactory().getSchema("TimeBasedTrkg::TBHits")); //NB: need to change to dc::hits (?)
    	echits = new Bank(reader.getSchemaFactory().getSchema("ECAL::hits"));
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
    	INDArray ECArray=Nd4j.zeros(6*batchSize,6,72,1);
    	int nPred=0;
    	//read all events until limit
    	while (reader.hasNext() == true && nPred<(batchSize*6)) {
        	reader.nextEvent(event);
        	event.read(dchits);
        	event.read(echits);
        	//Fill output arrays for each sector
        	for(int sector=1;sector<7;sector++) {
            	DCArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(FillDCArray(sector));
            	ECArray.get(NDArrayIndex.point(nPred), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(FillECArray(sector));		
            	nPred++;
            }
        }
    	//Creates list for output
    	INDArray[] output=new INDArray[2];
    	output[0]=DCArray;
    	output[1]=ECArray;
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
	 * 			EC image for a given sector.
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
	

	/**
	 *  Create EC images for a given sector from the echits bank.
	 *  
	 * Arguments:
	 *  		sector: sector for which to create the image
	 *  
	 * Returns:
	 * 			EC image for a given sector.
	 */
	private INDArray FillECArray(int sector) {
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
    
	 /**
	  *  Check if there's more events to be read.
	  *  
	  * Returns:
	  * 			True if the hipo reader still has more events to read.
	  */
    public boolean hasNext() {return reader.hasNext();}
    
    /**
	  *  Do something with the trigger model predictions
	  *  
	  * Arguments:
	  * 			resutls: INDArray containing the model predictions.
	  */
    public void apply(INDArray result) {
    	for(int event=0;event<batchSize;event++) {
        	for(int sector=0;sector<6;sector++) {
            	if(result.getFloat((event*6+sector),0)==1) {
                	System.out.println("Event "+event+" is predicted to have an electron in sector "+(sector+1));
                }
            }
        }
    }//END apply
    
}//EndOfClass
