package xenon.nn.layer;

import xenon.math.Tensor;
import xenon.nn.initialization.Initializer;

public class Softmax extends Layer {

    private int IN;

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN = inDimensions[1];
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, IN};
    }

    @Override
    public Tensor feedForward(Tensor tensor) {
        return softmax(tensor.get(1, new int[]{0, 0}));
    }

    private Tensor softmax(double[] in) {
        double sum = 0;
        double max = max(in);
        double[] expValues = new double[IN];
        for (int i = 0; i < IN; i++) {
            double value = Math.exp(in[i] - max);
            expValues[i] = value;
            sum += value;
        }
        for (int i = 0; i < IN; i++) {
            in[i] = expValues[i] / sum;
        }
        return new Tensor(in, 1, IN);
    }

    @Override
    public void initializeWeights() {

    }
    @Override
    public Tensor getDelta(Tensor outputError) {
        return outputError;
    }

    private static double max(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            max = Math.max(values[i], max);
        }
        return max;
    }
}
