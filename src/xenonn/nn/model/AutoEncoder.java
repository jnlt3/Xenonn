package xenonn.nn.model;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;
import xenonn.nn.layer.type.Trainable;

import java.util.Arrays;

public class AutoEncoder extends Model {

    private final int[] MODEL_INPUT_SHAPE;


    public AutoEncoder(int[] inputShape, Layer... layers) {
        MODEL_INPUT_SHAPE = inputShape;
        LAYERS = layers;
    }

    @Override
    public Tensor feedForward(Tensor Tensor) {
        Tensor out = Tensor.copy();
        for (int i = 0; i < LAYERS.length; i++) {
            out = LAYERS[i].feedForward(out);
        }
        return out;
    }

    public Tensor encode(Tensor Tensor, int encodingLayer) {
        Tensor out = Tensor.copy();
        for (int i = 0; i <= encodingLayer; i++) {
            out = LAYERS[i].feedForward(out);
        }
        return out;
    }


    @Override
    public void initialize() {
        LAYERS[0].setInDimensions(MODEL_INPUT_SHAPE);
        for (int i = 1; i < LAYERS.length; i++) {
            LAYERS[i].setInDimensions(LAYERS[i - 1].getOutDimensions());
        }
        for (int i = 0; i < LAYERS.length; i++) {
            LAYERS[i].initializeWeights();
        }
    }

    @Override
    public String summary() {
        final int LENGTH = 25;
        int numConnections = 0;
        String summary = "IN: " + " ".repeat(LENGTH - 4);
        summary += Arrays.toString(MODEL_INPUT_SHAPE) + " \n";
        for (int i = 0; i < LAYERS.length; i++) {
            int[] outputShape = LAYERS[i].getOutDimensions();
            String layerName = LAYERS[i].getClass().getSimpleName();
            int spaceNum = LENGTH - layerName.length();
            String spaces = " ".repeat(spaceNum - 1);
            summary += LAYERS[i].getClass().getSimpleName() + ":" + spaces + Arrays.toString(outputShape);
            if (LAYERS[i] instanceof Trainable) {
                int numCurrentConnections = 0;
                Trainable layer = (Trainable) LAYERS[i];
                Tensor[] weights = layer.getWeights();
                for (int j = 0; j < weights.length; j++) {
                    numCurrentConnections += weights[j].size();
                }
                numConnections += numCurrentConnections;
                summary += "  PARAMETERS: " + numCurrentConnections;
            }
            summary += "\n";
        }
        summary += "PARAMETERS TO TRAIN:  " + numConnections;
        return summary;
    }

}
