package xenon.nn.model;

import xenon.math.Tensor;
import xenon.nn.initialization.Initializer;
import xenon.nn.layer.Layer;

public abstract class Model {

    Layer[] LAYERS;

    public abstract Tensor feedForward(Tensor Tensor);

    public abstract void initialize();

    public void learn(boolean learn) {
        for (int i = 0; i < LAYERS.length; i++) {
            LAYERS[i].learning(learn);
        }
    }

    public abstract String summary();

    public Layer[] getLayers() {
        return LAYERS;
    }
}
