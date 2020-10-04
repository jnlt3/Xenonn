package xenonn.nn.layer;

import xenonn.math.Tensor;

public abstract class Layer {

    protected boolean learningOn;

    public void learning(boolean on) {
        this.learningOn = on;
    }

    public boolean learningOn() {
        return learningOn;
    }

    public abstract void setInDimensions(int[] inDimensions);

    public abstract int[] getOutDimensions();

    public abstract Tensor feedForward(Tensor in);

    public abstract Tensor getDelta(Tensor outputError);

    public abstract void initializeWeights();

}
