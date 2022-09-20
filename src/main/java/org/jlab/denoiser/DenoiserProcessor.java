/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.denoiser;

/**
 *
 * @author gavalian
 */
public interface DenoiserProcessor {    
    public void setThreshold(double threshold);
    public void initNetwork();
    public void processNext(InputDataStream stream);
}
