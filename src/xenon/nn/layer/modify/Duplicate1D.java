package xenon.nn.layer.modify;

import xenon.math.Tensor;
import xenon.nn.layer.Layer;

public class Duplicate1D extends Layer {

    private final int DUPLICATION_AMOUNT;

    private int[] inDimensions;
    private int[] outDimensions;
    private int inLength;
    private int outLength;

    public Duplicate1D(int duplicationAmount) {
        DUPLICATION_AMOUNT = duplicationAmount;
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        this.inDimensions = inDimensions;
        inLength = inDimensions[1];
        outDimensions = inDimensions.clone();
        outDimensions[1] *= DUPLICATION_AMOUNT;
        outLength = outDimensions[1];
    }

    @Override
    public int[] getOutDimensions() {
        return outDimensions;
    }

    @Override
    public Tensor feedForward(Tensor in) {
        Tensor out = Tensor.zeros(1, outLength);
        for (int i = 0; i < DUPLICATION_AMOUNT; i++) {
            for (int j = 0; j < inLength; j++) {
                int[] indices = new int[]{0, i * inLength + j};
                out.set(indices, in.get(0, j));
            }
        }
        return out;
    }



    @Override
    public Tensor getDelta(Tensor outputError) {
        Tensor delta = Tensor.zeros(inDimensions);
        for (int i = 0; i < DUPLICATION_AMOUNT; i++) {
            for (int j = 0; j < inLength; j++) {
                int[] indices = new int[]{0, j};
                delta.set(indices, delta.get(indices) + outputError.get(0, i * inLength + j));
            }
        }
        return delta;
    }

    @Override
    public void initializeWeights() {

    }
}
