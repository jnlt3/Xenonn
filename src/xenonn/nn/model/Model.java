package xenonn.nn.model;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;

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
