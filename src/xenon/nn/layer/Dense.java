package xenon.nn.layer;

import xenon.math.Tensor;
import xenon.nn.initialization.Initializer;
import xenon.nn.initialization.NullInitializer;
import xenon.nn.initialization.XavierInitializerNormal;
import xenon.nn.layer.type.Trainable;

public class Dense extends Layer implements Trainable {

    private int IN;
    private int OUT;

    private final Tensor[] WEIGHTS;

    private Tensor input;
    private Tensor outputError;

    private final Initializer INITIALIZER;

    public Dense(int out) {
        this(out, new XavierInitializerNormal());
    }

    public Dense(int out, Tensor[] weights) {
        this.WEIGHTS = weights;
        OUT = out;
        INITIALIZER = new NullInitializer();
    }

    public Dense(int out, Initializer initializer) {
        WEIGHTS = new Tensor[2];
        OUT = out;
        INITIALIZER = initializer;
    }

    public void initializeWeights() {
        INITIALIZER.initialize(WEIGHTS[0]);
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN = inDimensions[1];
        if (WEIGHTS[0] == null) {
            WEIGHTS[0] = Tensor.zeros(IN, OUT);
        }
        if (WEIGHTS[1] == null) {
            WEIGHTS[1] = Tensor.zeros(1, OUT);
        }
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, OUT};
    }

    @Override
    public Tensor feedForward(Tensor in) {
        if (learningOn) {
            input = in.copy();
        }
        Tensor value = Tensor.dotProduct(in, WEIGHTS[0]);
        value.add(WEIGHTS[1]);
        return value;
    }

    @Override
    public Tensor[] getWeights() {
        return WEIGHTS;
    }


    @Override
    public Tensor getDelta(Tensor outputError) {
        return Tensor.dotProduct(outputError, Tensor.transpose(WEIGHTS[0]));
    }

    @Override
    public int[][] weightShape() {
        return new int[][]{{IN, OUT}, {1, OUT}};
    }

    @Override
    public Tensor[] getGradients() {
        return new Tensor[]{Tensor.dotProduct(Tensor.transpose(input), outputError), outputError};
    }

    @Override
    public void setOutputError(Tensor outputError) {
        this.outputError = outputError.copy();
    }
}
