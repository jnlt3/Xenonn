package xenon.nn.learningrule;

import xenon.math.Tensor;
import xenon.nn.layer.Layer;
import xenon.nn.layer.type.Trainable;
import xenon.nn.loss.Loss;
import xenon.nn.model.Model;
import xenon.nn.optimizer.Optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BackPropagation extends GradientBasedOptimizer {


    private final Optimizer OPTIMIZER;

    private Tensor[][] weightGradients;

    public BackPropagation(Model model, Optimizer optimizer, Loss loss) {
        super(model, loss);
        OPTIMIZER = optimizer;
        weightGradients = new Tensor[model.getLayers().length][];
    }


    @Override
    public void trainIterations(int iterations, int batchSize) {
        MODEL.learn(true);
        if (trainIn.isEmpty()) {
            System.err.println("No training set");
        } else {
            for (int i = 0; i < iterations; i++) {
                for (int j = 0; j < batchSize; j++) {
                    int index = (int) (Math.random() * trainIn.size());
                    backPropagate(trainIn.get(index), trainOut.get(index));
                }
                step();
            }
        }
        MODEL.learn(false);
    }


    @Override
    public void trainEpochs(int epochs, int batchSize) {
        MODEL.learn(true);
        int size = trainIn.size();
        if (size == 0) {
            System.err.println("No training set");
        } else {
            for (int e = 0; e < epochs; e++) {
                List<Integer> indices = new ArrayList<>();
                for (int i = 0; i < trainIn.size(); i++) {
                    indices.add(i);
                }
                Collections.shuffle(indices);
                for (int i = 0; i < size; i += batchSize) {
                    int batchLeft = Math.min(size - i, batchSize);
                    BatchRunner main = new BatchRunner(indices, i, i + batchLeft);
                    main.run();
                    step();
                }
            }
        }
        MODEL.learn(false);
    }

    @Override
    public Tensor trainOnError(Tensor error, boolean returnInputError) {
        MODEL.learn(true);
        Tensor inputError = backPropagateOnError(error, returnInputError);
        MODEL.learn(false);
        return inputError;
    }

    private void backPropagate(Tensor in, Tensor expectedOut) {
        Tensor out = MODEL.feedForward(in);

        Tensor error = LOSS.getGradient(out, expectedOut);

        backPropagateOnError(error, false);
    }

    private Tensor backPropagateOnError(Tensor error, boolean returnInputError) {
        for (int i = NUM_LAYERS - 1; i >= 0; i--) {
            error = backPropagateThroughLayer(error, i, returnInputError);
        }
        return error;
    }

    private Tensor backPropagateThroughLayer(Tensor outputError, int layerIndex, boolean returnInputError) {
        Layer currentLayer = LAYERS[layerIndex];

        if (currentLayer instanceof Trainable) {
            Trainable trainableLayer = ((Trainable) currentLayer);
            trainableLayer.setOutputError(outputError);
            Tensor[] layerGradients = trainableLayer.getGradients();
            if (weightGradients[layerIndex] == null) {
                weightGradients[layerIndex] = layerGradients;
            } else {
                for (int j = 0; j < weightGradients[layerIndex].length; j++) {
                    weightGradients[layerIndex][j].add(layerGradients[j]);
                }
            }
        }
        return (layerIndex == 0 && !returnInputError) ? null : currentLayer.getDelta(outputError);
    }

    @Override
    public void step() {
        OPTIMIZER.step(weightGradients);
        for (int i = 0; i < weightGradients.length; i++) {
            if (weightGradients[i] != null) {
                int length = weightGradients[i].length;
                for (int j = 0; j < length; j++) {
                    weightGradients[i][j].zero();
                }
            }
        }
    }

    private class BatchRunner {

        private final List<Integer> INDICES;
        private final int START_INDEX;
        private final int END_INDEX;

        public BatchRunner(List<Integer> indices, int startIndex, int endIndex) {
            INDICES = indices;
            START_INDEX = startIndex;
            END_INDEX = endIndex;
        }

        public void run() {
            for (int i = START_INDEX; i < END_INDEX; i++) {
                int index = INDICES.get(i);
                backPropagate(trainIn.get(index), trainOut.get(index));
            }
        }
    }
}
