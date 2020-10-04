package xenon.nn.learningrule;

import xenon.math.Tensor;
import xenon.nn.layer.Layer;
import xenon.nn.layer.type.TrainableMemory;
import xenon.nn.layer.type.Trainable;
import xenon.nn.loss.Loss;
import xenon.nn.model.Model;
import xenon.nn.optimizer.Optimizer;

import java.util.ArrayList;
import java.util.List;

public class BackPropagationMemory extends GradientBasedOptimizer {


    private final Optimizer OPTIMIZER;

    private Tensor[][] weightGradients;

    private ArrayList<Tensor[]>[] memoryCells;
    private ArrayList<Tensor>[] inputs;

    private Tensor[][] initialMemory;
    private final int[] MEMORY_LAYER_INDICES;
    private final int MEMORY_LAYER_NUM;

    private int memoryIndex = -1;

    private final boolean TRAIN_MEMORY;
    private final boolean TRAIN_NORMAL;


    public BackPropagationMemory(Model model, Optimizer optimizer, Loss loss) {
        this(model, optimizer, loss, true, true);
    }

    public BackPropagationMemory(Model model, Optimizer optimizer, Loss loss, boolean trainMemory, boolean trainNormal) {
        super(model, loss);
        OPTIMIZER = optimizer;
        TRAIN_MEMORY = trainMemory;
        TRAIN_NORMAL = trainNormal;
        weightGradients = new Tensor[model.getLayers().length][];

        int memoryLayerNum = 0;
        List<Integer> memoryLayerIndices = new ArrayList<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (LAYERS[i] instanceof TrainableMemory) {
                memoryLayerNum++;
                memoryLayerIndices.add(i);
            }
        }
        MEMORY_LAYER_NUM = memoryLayerNum;
        memoryCells = new ArrayList[MEMORY_LAYER_NUM];
        inputs = new ArrayList[MEMORY_LAYER_NUM];
        MEMORY_LAYER_INDICES = new int[MEMORY_LAYER_NUM];
        for (int i = 0; i < MEMORY_LAYER_NUM; i++) {
            MEMORY_LAYER_INDICES[i] = memoryLayerIndices.get(i);
        }
    }

    @Override
    public void trainIterations(int iterations, int batchSize) {
        int size = trainIn.size();
        if (size == 0) {
            System.err.println("No training set");
        } else {
            saveMemory();
            MODEL.learn(true);
            resetTrain();
            for (int i = 0; i < iterations; i++) {
                int index = (int) (Math.random() * size);
                Tensor out = trainOut.get(index);
                if (out != null) {
                    backPropagate(trainIn.get(index), out, index);
                }
                if (i % batchSize == 0) {
                    step();
                }
            }
            step();
            resetCache();
            MODEL.learn(false);
        }
    }


    @Override
    public void trainEpochs(int epochs, int batchSize) {
        int size = trainIn.size();
        if (size == 0) {
            System.err.println("No training set");
        } else {
            saveMemory();
            MODEL.learn(true);
            resetTrain();
            for (int e = 0; e < epochs; e++) {
                for (int i = 0; i < size; i += batchSize) {
                    int batchLeft = Math.min(size - i, batchSize);
                    for (int j = 0; j < batchLeft; j++) {
                        int index = i + j;
                        Tensor out = trainOut.get(index);
                        if (out != null) {
                            backPropagate(trainIn.get(index), out, index);
                        }
                    }
                    step();
                }
            }
            resetCache();
            MODEL.learn(false);
        }
    }

    @Override
    public Tensor trainOnError(Tensor error, boolean returnInputError) {
        saveMemory();
        MODEL.learn(true);
        resetTrain();
        Tensor inputError = backPropagateOnError(error, trainIn.size() - 1, returnInputError);
        resetTrain();
        MODEL.learn(false);
        return inputError;
    }


    private void saveMemory() {
        initialMemory = new Tensor[MEMORY_LAYER_NUM][];
        for (int i = 0; i < MEMORY_LAYER_NUM; i++) {
            int layerIndex = MEMORY_LAYER_INDICES[i];
            initialMemory[i] = ((TrainableMemory) LAYERS[layerIndex]).getMemory();
        }
    }

    private void resetTrain() {
        resetCache();
        resetMemory();
        for (int i = 0; i < trainIn.size(); i++) {
            addMemoryCell();
            MODEL.feedForward(trainIn.get(i));
            addInput();
        }
        addMemoryCell();
        resetMemory();
    }

    private void resetCache() {
        memoryCells = new ArrayList[MEMORY_LAYER_NUM];
        inputs = new ArrayList[MEMORY_LAYER_NUM];
        for (int i = 0; i < MEMORY_LAYER_NUM; i++) {
            memoryCells[i] = new ArrayList<>();
            inputs[i] = new ArrayList<>();
        }
    }

    private void resetMemory() {
        for (int i = 0; i < MEMORY_LAYER_NUM; i++) {
            int layerIndex = MEMORY_LAYER_INDICES[i];
            TrainableMemory memoryLayer = (TrainableMemory) LAYERS[layerIndex];
            memoryLayer.setMemory(initialMemory[i]);
        }
    }

    private void addMemoryCell() {
        int memoryIndex = 0;
        for (int i = 0; i < MEMORY_LAYER_NUM; i++) {
            int layerIndex = MEMORY_LAYER_INDICES[i];
            TrainableMemory memoryLayer = (TrainableMemory) LAYERS[layerIndex];
            memoryCells[memoryIndex].add(memoryLayer.getMemory());
            memoryIndex++;
        }
    }

    private void addInput() {
        int memoryIndex = 0;
        for (int i = 0; i < MEMORY_LAYER_NUM; i++) {
            int layerIndex = MEMORY_LAYER_INDICES[i];
            TrainableMemory memoryLayer = (TrainableMemory) LAYERS[layerIndex];
            inputs[memoryIndex].add(memoryLayer.getInput());
            memoryIndex++;
        }
    }

    private void setMemory(int index) {
        int memoryIndex = 0;
        for (int i = 0; i < MEMORY_LAYER_NUM; i++) {
            int layerIndex = MEMORY_LAYER_INDICES[i];
            TrainableMemory memoryLayer = (TrainableMemory) LAYERS[layerIndex];
            memoryLayer.setMemory(memoryCells[memoryIndex].get(index));
            memoryLayer.setNextMemory(memoryCells[memoryIndex].get(index + 1));
            memoryLayer.setInput(inputs[memoryIndex].get(index));
            memoryIndex++;
        }
    }

    private void backPropagate(Tensor in, Tensor expectedOut, int index) {
        setMemory(index);
        Tensor out = MODEL.feedForward(in);
        Tensor error = LOSS.getGradient(out, expectedOut);
        backPropagateOnError(error, index, false);
    }

    private Tensor backPropagateOnError(Tensor error, int index, boolean returnInputError) {
        memoryIndex = MEMORY_LAYER_NUM - 1;
        for (int i = NUM_LAYERS - 1; i >= 0; i--) {
            error = backPropagateThroughLayer(error, i, index, returnInputError);
        }
        return error;
    }

    private Tensor backPropagateThroughLayer(Tensor outputError, int layerIndex, int cacheIndex, boolean returnInputError) {
        Layer layer = LAYERS[layerIndex];
        if (layer instanceof Trainable) {
            Trainable trainableLayer = ((Trainable) layer);

            trainableLayer.setOutputError(outputError);
            if (TRAIN_MEMORY && trainableLayer instanceof TrainableMemory) {
                backPropagateThroughTime(outputError, memoryIndex, cacheIndex);
                memoryIndex--;
            } else if (TRAIN_NORMAL) {
                Tensor[] layerGradients = trainableLayer.getGradients();
                if (weightGradients[layerIndex] == null) {
                    weightGradients[layerIndex] = layerGradients;
                } else {
                    for (int j = 0; j < weightGradients[layerIndex].length; j++) {
                        weightGradients[layerIndex][j].add(layerGradients[j]);
                    }
                }
            }
        }
        return (layerIndex == 0 && !returnInputError) ? null : layer.getDelta(outputError);
    }

    private void backPropagateThroughTime(Tensor memoryError, int memoryIndex, int cacheIndex) {
        int layerIndex = MEMORY_LAYER_INDICES[memoryIndex];
        TrainableMemory memoryLayer = (TrainableMemory) LAYERS[layerIndex];
        Tensor error = memoryError.copy();
        for (int j = cacheIndex; j >= 0; j--) {
            memoryLayer.setMemory(memoryCells[memoryIndex].get(j));
            memoryLayer.setNextMemory(memoryCells[memoryIndex].get(j + 1));
            memoryLayer.setInput(inputs[memoryIndex].get(j));
            memoryLayer.calculateGradients(error);
            if (j != 0) {
                error = memoryLayer.getMemoryDelta(error);
            }
        }
        Tensor[] layerGradients = memoryLayer.getGradients();
        if (weightGradients[layerIndex] == null) {
            weightGradients[layerIndex] = layerGradients;
        } else {
            for (int j = 0; j < weightGradients[layerIndex].length; j++) {
                weightGradients[layerIndex][j].add(layerGradients[j]);
            }
        }
        memoryLayer.setMemory(memoryCells[memoryIndex].get(cacheIndex));
        memoryLayer.setNextMemory(memoryCells[memoryIndex].get(cacheIndex + 1));
        memoryLayer.setInput(inputs[memoryIndex].get(cacheIndex));
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
        resetTrain();
    }
}
