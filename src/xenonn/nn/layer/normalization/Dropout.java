package xenonn.nn.layer.normalization;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;

public class Dropout extends Layer {

    private int[] IN_DIMENSIONS;
    private int IN;
    private final double DROPOUT;
    private transient Tensor derivatives;

    public Dropout(double dropout) {
        DROPOUT = dropout;
    }

    public void initializeWeights() {

    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN_DIMENSIONS = inDimensions;
        IN = 1;
        for (int i = 0; i < IN_DIMENSIONS.length; i++) {
            IN *= IN_DIMENSIONS[i];
        }
    }

    @Override
    public int[] getOutDimensions() {
        return IN_DIMENSIONS;
    }

    @Override
    public Tensor feedForward(Tensor tensor) {
        if (learningOn) {
            Tensor out = tensor.copy();
            out.reshape(1, IN);
            derivatives = Tensor.ones(1, IN);
            for (int i = 0; i < IN; i++) {
                int[] indices = new int[]{0, i};
                if (Math.random() < DROPOUT) {
                    out.set(indices, 0);
                    derivatives.set(indices, 0);
                }
            }
            out.reshape(tensor.getShape());
            return out;
        }
        return tensor;
    }



    @Override
    public Tensor getDelta(Tensor outputError) {
        Tensor delta = outputError.copy();
        delta.reshape(1, IN);
        delta.multiply(derivatives);
        delta.reshape(outputError.getShape());
        return delta;
    }
}
