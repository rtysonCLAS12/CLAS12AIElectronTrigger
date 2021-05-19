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
public class HipoInputDataStream implements InputDataStream {
    HipoReader reader = null;
    public void open(String url);
    public void setBatch(int size);
    public List<INDArray> next();
    public boolean hasNext();
    public void apply(List<INDArray> result);
}
