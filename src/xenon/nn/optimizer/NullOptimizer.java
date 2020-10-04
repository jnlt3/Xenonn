package xenon.nn.optimizer;

import xenon.math.Tensor;
import xenon.nn.model.Model;

public class NullOptimizer extends Optimizer{

    public NullOptimizer() {
        super(null);
    }

    @Override
    public void step(Tensor[][] gradients) {

    }
}
