package xenon.nn.activation;

public class Spike extends ActivationFunction {

    public double value(double in) {
        return Math.max(1 - Math.abs(in), 0);
    }

    public double derivative(double in) {
        return (in < -1 || in > 1) ? 0 : -Math.signum(in);
    }
}
