package xenonn.nn.layer;

import xenonn.math.Tensor;

public class Flatten extends Layer {

    private int[] IN_DIMENSIONS;
    private int OUT;

    public Flatten() {

    }

    public void initializeWeights() {

    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN_DIMENSIONS = inDimensions;
        int numInputs = 1;
        for (int i = 0; i < IN_DIMENSIONS.length; i++) {
            numInputs *= IN_DIMENSIONS[i];
        }
        OUT = numInputs;
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, OUT};
    }

    @Override
    public Tensor feedForward(Tensor tensor) {
        Tensor flattened = tensor.copy();
        flattened.reshape(1, tensor.size());
        return flattened;
    }



    @Override
    public Tensor getDelta(Tensor outputError) {
        outputError.reshape(IN_DIMENSIONS);
        return outputError;
    }
}
