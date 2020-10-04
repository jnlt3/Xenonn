package xenonn.nn.layer.modify;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;
import xenonn.nn.layer.type.Trainable;

public class Scaler extends Layer implements Trainable {


    private int[] outDimensions;
    private int[][] weightShape = {{1}};
    private Tensor[] weights = new Tensor[1];

    private Tensor[] gradients = new Tensor[1];

    private Tensor input;

    private transient Tensor outputError;

    public void initializeWeights() {
        weights[0] = Tensor.ones(1);
        gradients[0] = Tensor.zeros(1);
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        outDimensions = inDimensions.clone();
    }

    @Override
    public int[] getOutDimensions() {
        return outDimensions;
    }

    @Override
    public Tensor feedForward(Tensor in) {
        if (learningOn) {
            input = in.copy();
        }
        Tensor out = Tensor.multiply(in, weights[0].get(0));
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
        gradients[0].set(new int[1], Tensor.multiply(input, outputError.get(0)).sum());
        return gradients;
    }

    @Override
    public void setOutputError(Tensor outputError) {
        this.outputError = outputError;
    }
}
