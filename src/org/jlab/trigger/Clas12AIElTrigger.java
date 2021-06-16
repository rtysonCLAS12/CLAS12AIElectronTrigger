package org.jlab.trigger;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Clas12AIElTrigger {
	/*
	 * Loads the AI Trigger classifier from hardcoded location.
	 */
	public void LoadNetwork();
	
	/*
	 * Returns the predictions made by the AI Trigger classifier on data loaded and prepared by parseData.
	 * Predictions are made per batches.
	 * Will give an exception if the network isn't loaded correctly.
	 * 
	 * Arguments:
	 * 			BatchSize: Size of batch of predictions, number of predictions must be divisible by BatchSize!!
	 * 
	 * Returns:
	 * 			The response of the classifier as an NPrediction*2 INDArray with the probability that
	 * an electron is in that sector in the first column, and the probability that there isn't an electron
	 * in that sector in the second column.
	 * 	 
	 */
	public INDArray Predict(int BatchSize);
	
	/*
	 * Applies a threshold to the classifier output (response). This varies from 0 to 1, and so we
	 * round this to 1 (0) if the response is above (equal or below) the threshold.
	 * 
	 * Arguments:
	 * 			Predictions: An INDArray containing the classifier output. The is the probability that
	 *  an event is of the positive sample in column 0, and of the negative sample in column 1. 
	 * 			Threshold: The desired threshold on the response.
	 * 
	 * Returns:
	 * 			The response of the classifier rounded based on the inputed threshold.
	 * 	 
	 */
	public int[] ApplyResponseThreshold(INDArray Predictions, double Threshold);
	
	/*
	 * Returns the data parsed into the correct format for the AI Trigger classifier.
	 *
	 * 
	 * Returns:
	 * 			INDArray List containing the DC images in Data[0], and EC images in Data[1]
	 * 	 
	 */
	public INDArray[] ParseData();

}
