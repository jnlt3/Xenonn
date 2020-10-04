package xenon.nn.initialization;

import xenon.math.Tensor;

import java.util.Random;

public class XavierInitializerUniform extends Initializer {

    private static final double SQRT_6 = Math.sqrt(6);

    @Override
    public void initialize(Tensor weightTensor) {
        int rows = weightTensor.getShape()[0];
        int columns = weightTensor.getShape()[1];
        double range = SQRT_6 / Math.sqrt(rows + columns);

        double[] values = weightTensor.getAll();
        for (int i = 0; i < values.length; i++) {
            values[i] = RANDOM.nextGaussian() * 2 * range - range;
        }
    }
}
