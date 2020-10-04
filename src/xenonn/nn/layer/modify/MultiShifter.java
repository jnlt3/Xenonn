package xenonn.nn.layer.modify;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;
import xenonn.nn.layer.type.Trainable;

public class MultiShifter extends Layer implements Trainable {


    private int[] inDimensions;
    private int[] outDimensions;
    private int[][] weightShape = new int[1][];
    private Tensor[] weights = new Tensor[1];

    private transient Tensor outputError;

    public void initializeWeights() {
        weights[0] = Tensor.zeros(inDimensions);
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        this.inDimensions = inDimensions;
        outDimensions = inDimensions.clone();
        weightShape[0] = inDimensions;
    }

    @Override
    public int[] getOutDimensions() {
        return outDimensions;
    }

    @Override
    public Tensor feedForward(Tensor in) {
        Tensor out = Tensor.add(in, weights[0]);
        return out;
    }

    @Override
    public Tensor[] getWeights() {
        return weights;
    }



    @Override
    public Tensor getDelta(Tensor outputError) {
        return outputError;
    }

    @Override
    public int[][] weightShape() {
        return weightShape;
    }

    @Override
    public Tensor[] getGradients() {
        return new Tensor[]{outputError};
    }

    @Override
    public void setOutputError(Tensor outputError) {
        this.outputError = outputError;
    }
}
