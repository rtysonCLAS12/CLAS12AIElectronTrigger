/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.trigger;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public interface TriggerProcessor {    
    public void setThreshold(double threshold);
    public void initNetwork(String url);
    public void processNext(InputDataStream stream,OutputDataStream outStream);
    public void apply(OutputDataStream outStream,INDArray result);
}
