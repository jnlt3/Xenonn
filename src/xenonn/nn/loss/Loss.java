package xenonn.nn.loss;

import xenonn.math.Tensor;

import java.util.List;

public abstract class Loss {

    public double getLoss(List<Tensor> out, List<Tensor> expectedOut) {
        if (out.isEmpty()) {
            return 0;
        }
        double loss = 0;
        for (int i = 0; i < out.size(); i++) {
            loss += getLoss(out.get(i), expectedOut.get(i)) / out.size();
        }
        return loss;
    }

    public abstract double getLoss(Tensor out, Tensor expectedOut);

    public abstract Tensor getGradient(Tensor out, Tensor expectedOut);
}
