package xenon.nn.layer;

import xenon.math.Tensor;
import xenon.nn.activation.ActivationFunction;

import java.util.Arrays;

public class Activation extends Layer {

    private final ActivationFunction ACTIVATION_FUNCTION;


    private int[] IN_DIMENSIONS;
    private int IN;

    private Tensor derivatives;

    public Activation(ActivationFunction activationFunction) {
        ACTIVATION_FUNCTION = activationFunction;
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
            derivatives = ACTIVATION_FUNCTION.derivative(tensor);
        }
        return ACTIVATION_FUNCTION.value(tensor);
    }

    @Override
    public Tensor getDelta(Tensor outputError) {
        return Tensor.multiply(outputError, derivatives);
    }

}
