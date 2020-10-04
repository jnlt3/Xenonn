package xenon.nn.loss;

import xenon.math.Tensor;

import java.util.List;

public class CrossEntropyLoss extends Loss {

    private static final int[] INDICES = new int[2];

    @Override
    public double getLoss(Tensor out, Tensor expectedOut) {
        int[] indices = new int[2];
        double[] output = out.get(1, indices);
        double[] expectedOutput = expectedOut.get(1, indices);
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            if (expectedOutput[i] == 1) {
                loss += -Math.log(output[i]);
            } else {
                loss += -Math.log(1 - output[i]);
            }
        }
        return loss;
    }

    @Override
    public Tensor getGradient(Tensor out, Tensor expectedOut) {
        Tensor grad = out.copy();
        int label = argmax(expectedOut.get(1, INDICES));
        grad.set(new int[]{0, label}, grad.get(0, label) - 1);
        grad.multiply(1D / out.getShape()[1]);
        return grad;
    }

    private static final int argmax(double[] d) {
        int index = 0;
        for (int i = 1; i < d.length; i++) {
            if (d[i] > d[index]) {
                index = i;
            }
        }
        return index;
    }
}
