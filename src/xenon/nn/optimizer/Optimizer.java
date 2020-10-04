package xenon.nn.optimizer;

import xenon.math.Tensor;
import xenon.nn.model.Model;

public abstract class Optimizer {

    protected final Model MODEL;

    public Optimizer(Model model) {
        MODEL = model;
    }

    public abstract void step(Tensor[][] gradients);
}
