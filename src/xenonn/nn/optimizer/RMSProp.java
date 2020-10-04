package xenonn.nn.optimizer;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;
import xenonn.nn.layer.type.Trainable;
import xenonn.nn.model.Model;

public class RMSProp extends Optimizer {

    private static final double EPSILON = 1e-8;
    private final double LEARNING_RATE;
    private final double BETA;

    private final Tensor[][] CACHE;

    public RMSProp(Model model, double learningRate, double beta) {
        super(model);
        LEARNING_RATE = learningRate;
        BETA = beta;
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
                    CACHE[i][j].multiply(BETA);
                    CACHE[i][j].add(Tensor.multiply(Tensor.square(gradients[i][j]), 1 - BETA));
                    layer.getWeights()[j].subtract(Tensor.multiply(Tensor.divide(gradients[i][j], Tensor.add(Tensor.sqrt(CACHE[i][j]), EPSILON)), LEARNING_RATE));
                }
            }
        }
    }
}
