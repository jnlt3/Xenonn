package xenon.nn.optimizer;

import xenon.math.Tensor;
import xenon.nn.layer.Layer;
import xenon.nn.layer.type.Trainable;
import xenon.nn.model.Model;

public class SGD extends Optimizer {

    private final double LEARNING_RATE;
    private final double MOMENTUM;

    private final Tensor[][] CACHE;

    public SGD(Model model, double learningRate, double momentum) {
        super(model);
        LEARNING_RATE = learningRate;
        MOMENTUM = momentum;
        Layer[] layers = MODEL.getLayers();
        CACHE = new Tensor[layers.length][];
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof Trainable) {
                Trainable layer = (Trainable) layers[i];
                int[][] weightShape = layer.weightShape();
                int cacheLength = weightShape.length;
                CACHE[i] = new Tensor[cacheLength];
                for (int j = 0; j < cacheLength; j++) {
                    CACHE[i][j] = Tensor.zeros(weightShape[j]);
                }
            }
        }
    }

    @Override
    public void step(Tensor[][] gradients) {
        Layer[] layers = MODEL.getLayers();
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof Trainable && gradients[i] != null) {
                int length = gradients[i].length;
                Trainable layer = (Trainable) layers[i];
                for (int j = 0; j < length; j++) {
                    CACHE[i][j].multiply(MOMENTUM);
                    CACHE[i][j].add(Tensor.multiply(gradients[i][j], 1 - MOMENTUM));
                    layer.getWeights()[j].subtract(Tensor.multiply(CACHE[i][j], LEARNING_RATE));
                }
            }
        }
    }
}
