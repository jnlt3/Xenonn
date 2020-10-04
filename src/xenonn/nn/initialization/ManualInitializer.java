package xenonn.nn.initialization;

import xenonn.math.Tensor;

public class ManualInitializer extends Initializer {

    private final double RANGE;

    public ManualInitializer(double range) {
        RANGE = range;
    }

    @Override
    public void initialize(Tensor weightTensor) {
        double[] values = weightTensor.getAll();
        for (int i = 0; i < values.length; i++) {
            values[i] = RANDOM.nextGaussian() * RANGE;
        }
    }
}
