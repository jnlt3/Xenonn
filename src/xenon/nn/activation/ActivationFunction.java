package xenon.nn.activation;

import xenon.math.Tensor;

import java.util.Arrays;

public abstract class ActivationFunction {

    public Tensor value(Tensor in) {
        Tensor out = in.copy();
        double[] values = out.getAll();
        for (int i = 0; i < values.length; i++) {
            values[i] = value(values[i]);
        }
        return out;
    }

    public Tensor derivative(Tensor in) {
        Tensor out = in.copy();
        double[] derivatives = out.getAll();
        for (int i = 0; i < derivatives.length; i++) {
            derivatives[i] = derivative(derivatives[i]);
        }
        return out;
    }

    protected abstract double value(double in);

    protected abstract double derivative(double in);
}
