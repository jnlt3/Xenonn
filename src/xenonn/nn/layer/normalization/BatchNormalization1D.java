package xenonn.nn.layer.normalization;

import xenonn.math.Tensor;
import xenonn.nn.initialization.Initializer;
import xenonn.nn.initialization.XavierInitializerUniform;
import xenonn.nn.layer.Layer;

public class BatchNormalization1D extends Layer {

    private static final double EPSILON = 1e-4;

    private int[] IN_SHAPE;
    private int IN_SIZE;

    private final double MOMENTUM;

    private double averageMean;
    private double averageVariance = 1;

    private transient Tensor lastCenteredOutput;
    private transient double lastMean;
    private transient double lastVariance;
    private transient double lastStdDev;

    /**
     * This class doesn't work
     *
     * @param momentum
     */
    //TODO: Make this class work
    public BatchNormalization1D(double momentum) {
        this(momentum, new XavierInitializerUniform());
    }

    /**
     * This class doesn't work
     *
     * @param momentum
     */
    public BatchNormalization1D(double momentum, Initializer initializer) {
        MOMENTUM = momentum;
    }

    public void initializeWeights() {

    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN_SHAPE = inDimensions;
        IN_SIZE = 1;
        for (int i = 0; i < IN_SHAPE.length; i++) {
            IN_SIZE *= IN_SHAPE[i];
        }
    }

    @Override
    public int[] getOutDimensions() {
        return IN_SHAPE;
    }

    @Override
    public Tensor feedForward(Tensor in) {
        double mean = in.mean();
        double variance = in.variance();
        averageMean = averageMean * MOMENTUM + mean * (1 - MOMENTUM);
        averageVariance = averageVariance * MOMENTUM + variance * (1 - MOMENTUM);

        Tensor value = in.copy();
        value.reshape(1, IN_SIZE);
        if (learningOn) {
            lastVariance = variance;
            lastStdDev = (Math.sqrt(variance) + EPSILON);
            lastMean = mean;
            value.add(-mean);
            lastCenteredOutput = value.copy();
            value = Tensor.multiply(value, 1 / lastStdDev);
        } else {
            value = Tensor.multiply(Tensor.add(in, -averageMean), 1 / (Math.sqrt(averageVariance) + EPSILON));
        }
        value.reshape(IN_SHAPE);
        return value;
    }



    @Override
    public Tensor getDelta(Tensor outputError) {

        outputError.reshape(1, IN_SIZE);

        double invertedStdDev = 1 / lastStdDev;

        double sum = lastCenteredOutput.sum();
        double stdDevDerivative = sum * -0.5 * Math.pow(lastVariance, -1.5);
        double meanDerivative = -invertedStdDev + stdDevDerivative * sum;


        Tensor derivative0 = Tensor.multiply(outputError, invertedStdDev);
        Tensor derivative1 = Tensor.multiply(Tensor.add(outputError, -lastMean), 2);
        double derivative2 = meanDerivative * invertedStdDev / IN_SIZE;
        Tensor delta = Tensor.add(Tensor.add(derivative0, derivative1), derivative2);
        delta.reshape(IN_SHAPE);
        return delta;
    }

}
