/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.trigger;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @authors gavalian, tyson
 */
public interface InputDataStream {
    public void open(String url);
    public void setBatch(int size);
    public boolean hasNext();
    public INDArray[] next();
    public void apply(INDArray result);
}
