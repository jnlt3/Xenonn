package xenon.nn.optimizer;

import xenon.math.Tensor;
import xenon.nn.layer.Layer;
import xenon.nn.layer.type.Trainable;
import xenon.nn.model.Model;

import java.util.Arrays;

public class Adam extends Optimizer {

    private static final double EPSILON = 1e-8;
    private final double LEARNING_RATE;
    private final double BETA_0;
    private final double BETA_1;

    private final Tensor[][] CACHE_0;
    private final Tensor[][] CACHE_1;

    public Adam(Model model, double learningRate, double beta0, double beta1) {
        super(model);
        LEARNING_RATE = learningRate;
        BETA_0 = beta0;
        BETA_1 = beta1;
        Layer[] layers = MODEL.getLayers();
        CACHE_0 = new Tensor[layers.length][];
        CACHE_1 = new Tensor[layers.length][];
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof Trainable) {
                Trainable layer = (Trainable) layers[i];
                int[][] weightShape = layer.weightShape();
                int cacheLength = weightShape.length;
                CACHE_0[i] = new Tensor[cacheLength];
                CACHE_1[i] = new Tensor[cacheLength];
                for (int j = 0; j < cacheLength; j++) {
                    CACHE_0[i][j] = Tensor.zeros(weightShape[j]);
                    CACHE_1[i][j] = Tensor.zeros(weightShape[j]);
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
                    CACHE_0[i][j].multiply(BETA_0);
                    CACHE_0[i][j].add(Tensor.multiply(gradients[i][j], 1 - BETA_0));

                    CACHE_1[i][j].multiply(BETA_1);
                    CACHE_1[i][j].add(Tensor.multiply(Tensor.square(gradients[i][j]), 1 - BETA_1));

                    Tensor cache0Hat = Tensor.multiply(CACHE_0[i][j], 1 / (1 - BETA_0));
                    Tensor cache1Hat = Tensor.multiply(CACHE_1[i][j], 1 / (1 - BETA_1));
                    layer.getWeights()[j].subtract(Tensor.multiply(Tensor.divide(cache0Hat, Tensor.add(Tensor.sqrt(cache1Hat), EPSILON)), LEARNING_RATE));
                }
            }
        }
    }
}
