/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.trigger;

import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoWriter;
import org.jlab.jnp.hipo4.data.Schema;
import org.jlab.jnp.hipo4.data.Schema.SchemaBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @authors gavalian, tyson
 */
public class HipoOutputDataStream implements OutputDataStream {
    Bank bank = null;
    Event event = null;
    HipoWriter writer=null;

     /**
	 *  Opens a .hipo file at the specified location
	 *  
	 * Arguments:
	 *  		url: location of the hipo file
	 *  
	 */
    public void open(String url) {
    	writer.open(url);
	event = new Event();
    }

    /**
	 *  Closes previously opened .hipo file
	 *  
	 */
    public void close(){
        writer.close();
    }

    /**
     * Creates the hipo bank with a given structure
     *
     */
    public void createBank(){
	//not sure what 1 or 1 mean...
	SchemaBuilder builder = new SchemaBuilder("AITrigger::Output",1,1);

	builder.addEntry("event_number","I","")
	    .addEntry("response_s1","F","")
	    .addEntry("response_s2","F","")
	    .addEntry("response_s3","F","")
	    .addEntry("response_s4","F","")
	    .addEntry("response_s5","F","")
	    .addEntry("response_s6","F","");

	Schema sch = builder.build();
	sch.show();
	bank=new Bank(sch,1);

	writer = new HipoWriter();
	writer.getSchemaFactory().addSchema(sch);

    }

    public void output(INDArray result){

	for (int i = 0; i < result.shape()[0]; i++) {
            //Clear the content of the event, ready to write banks
	    event.reset(); 

            //Get and write the event number for each event
            int eventNumber = result.getInt(i,0);
	    bank.putInt("event_number", 0, eventNumber);

	    //loop over sectors, get and write response in each sector
	    for (Integer sector=1;sector<7;sector++) {
		float response=result.getFloat(i,sector);
		bank.putFloat("response_s"+sector.toString(), 0, response);
	    }
            

            // Add the bank to the event
            event.write(bank);

            // Write the event to the HipoWriter
            writer.addEvent(event);
        }

    }

}
