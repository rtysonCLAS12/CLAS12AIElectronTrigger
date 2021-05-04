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
	 * Returns the data parsed into the correct format for the AI Trigger classifier.
	 *
	 * 
	 * Returns:
	 * 			INDArray List containing the DC images in Data[0], and EC images in Data[1]
	 * 	 
	 */
	public INDArray[] ParseData();

}
