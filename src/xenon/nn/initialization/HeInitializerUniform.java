package xenon.nn.initialization;

import xenon.math.Tensor;

import java.util.Random;

public class HeInitializerUniform extends Initializer {

    @Override
    public void initialize(Tensor weightTensor) {
        int rows = weightTensor.getShape()[0];
        int columns = weightTensor.getShape()[1];
        double range = 2 / Math.sqrt(rows * columns);

        double[] values = weightTensor.getAll();
        for (int i = 0; i < values.length; i++) {
            values[i] = RANDOM.nextDouble() * 2 * range - range;
        }
    }
}
