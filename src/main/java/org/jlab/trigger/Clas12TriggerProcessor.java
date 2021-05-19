/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.trigger;

/**
 *
 * @author gavalian
 */
public class Clas12TriggerProcessor implements TriggerProcessor {
    
    double inferenceThreshold = 0.5;
    
    public void setThreshold(double threshold){ inferenceThreshold = threshold; }
    public void initNetwork(){
        // TODO -- initialize the network
    }
    
    public void processNext(InputDataStream stream){
        List<INDArray> array = stream.next();
        List<INDArray> result = null;// this inferenc will be done by network
        stream.apply(result);
    }
    
    public static void main(String[] args){
        Clas12TriggerProcessor processor = new Clas12TriggerProcessor();
        processor.initNetwork();
        
        HipoInputDataStream stream = new HipoInputDataStream();
        
        steam.open("data.hipo");
        
        while(stream.hasNext()){
            processor.processNext(stream);            
        }
        
    }
}
