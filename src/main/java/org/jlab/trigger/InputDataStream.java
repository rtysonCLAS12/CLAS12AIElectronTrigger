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
public interface InputDataStream {
    public void open(String url);
    public void setBatch(int size);
    public boolean hasNext();
    public List<INDArray> next();
    public void apply(List<INDArray> result);
}
