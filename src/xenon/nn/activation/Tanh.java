package xenon.nn.activation;


public class Tanh extends ActivationFunction {

    public double value(double in) {
        return Math.tanh(in);
    }

    public double derivative(double in) {
        return 2 / (Math.cosh(2 * in) + 1);
    }
}
