package xenonn.nn.optimizer;

import xenonn.math.Tensor;

public class NullOptimizer extends Optimizer{

    public NullOptimizer() {
        super(null);
    }

    @Override
    public void step(Tensor[][] gradients) {

    }
}
