package xenonn.nn.activation;

public class ReLU extends ActivationFunction {

    public double value(double in) {
        return Math.max(0, in);
    }

    public double derivative(double in) {
        return in >= 0 ? 1: 0;
    }

}
