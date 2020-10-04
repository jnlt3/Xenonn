package xenonn.nn.activation;


public class Sigmoid extends ActivationFunction {

    public double value(double in) {
        return 1 / (1 + Math.exp(-in));
    }

    public double derivative(double in) {
        in = value(in);
        return in * (1 - in);
    }
}
