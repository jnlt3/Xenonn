package xenon.nn.activation;

public class Linear extends ActivationFunction {

    private final double SHIFT;
    private final double SCALE;

    public Linear() {
        this(0, 1);
    }

    public Linear(double shift, double scale) {
        SHIFT = shift;
        SCALE = scale;
    }

    public double value(double in) {
        return SHIFT + in * SCALE;
    }

    public double derivative(double in) {
        return SCALE;
    }

}
