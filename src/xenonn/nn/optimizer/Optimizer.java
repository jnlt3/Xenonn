package xenonn.nn.optimizer;

import xenonn.math.Tensor;
import xenonn.nn.model.Model;

public abstract class Optimizer {

    protected final Model MODEL;

    public Optimizer(Model model) {
        MODEL = model;
    }

    public abstract void step(Tensor[][] gradients);
}
