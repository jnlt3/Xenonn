package xenon.nn.loss;

import xenon.math.Tensor;

import java.util.List;

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
