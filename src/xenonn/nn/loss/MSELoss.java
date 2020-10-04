package xenonn.nn.loss;

import xenonn.math.Tensor;

public class MSELoss extends Loss {

    @Override
    public double getLoss(Tensor out, Tensor expectedOut) {
        return Tensor.square(Tensor.subtract(out, expectedOut)).sum() / 2;
    }

    @Override
    public Tensor getGradient(Tensor out, Tensor expectedOut) {
        return Tensor.subtract(out, expectedOut);
    }
}
