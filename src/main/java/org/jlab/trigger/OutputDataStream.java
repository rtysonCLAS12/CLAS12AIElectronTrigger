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

public interface OutputDataStream {
    public void open(String url);
    public void output(INDArray result);
    public void createBank();

}
