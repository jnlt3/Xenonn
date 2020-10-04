package xenonn.nn.activation;

public class Gaussian extends ActivationFunction {

    public double value(double in) {
        return Math.exp(-in * in);
    }

    public double derivative(double in) {
        return -2 * in * value(in);
    }
}
