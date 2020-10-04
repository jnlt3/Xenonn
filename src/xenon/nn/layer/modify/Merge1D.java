package xenon.nn.layer.modify;

import xenon.math.Tensor;
import xenon.nn.layer.Layer;
import xenon.nn.layer.type.Trainable;
import xenon.nn.layer.type.TrainableMemory;

import java.util.Arrays;

public class Merge1D extends Layer implements Trainable, TrainableMemory {

    private final Layer LAYER_0;
    private final Layer LAYER_1;
    private int LAYER_0_INPUT;
    private int LAYER_0_OUTPUT;
    private int LAYER_1_OUTPUT;
    private int[] OUT_DIMENSIONS;

    int[] layer0Input;
    int[] layer1Input;
    int[] layer0Output;
    int[] layer1Output;

    private final boolean LAYER_0_TRAINABLE;
    private final boolean LAYER_1_TRAINABLE;
    private final boolean TRAINABLE;
    private final boolean LAYER_0_TRAINABLE_MEMORY;
    private final boolean LAYER_1_TRAINABLE_MEMORY;
    private final boolean TRAINABLE_MEMORY;

    private Trainable trainableLayer0 = null;
    private Trainable trainableLayer1 = null;
    private TrainableMemory trainableMemoryLayer0 = null;
    private TrainableMemory trainableMemoryLayer1 = null;

    int length = 0;
    int layer0WeightLength = 0;
    int layer1WeightLength = 0;

    int layer0Memory = 0;
    int layer1Memory = 0;
    int layer0MemoryLength = 0;
    int layer1MemoryLength = 0;

    int layer0InputLength = 0;
    int layer1InputLength = 0;

    private int[][] weightShape;
    private Tensor[] weights;

    public Merge1D(Layer layer0, Layer layer1, int layer0Input) {
        LAYER_0 = layer0;
        LAYER_1 = layer1;
        LAYER_0_INPUT = layer0Input;

        LAYER_0_TRAINABLE = LAYER_0 instanceof Trainable;
        LAYER_0_TRAINABLE_MEMORY = LAYER_0_TRAINABLE && LAYER_0 instanceof TrainableMemory;

        LAYER_1_TRAINABLE = LAYER_1 instanceof Trainable;
        LAYER_1_TRAINABLE_MEMORY = LAYER_1_TRAINABLE && LAYER_1 instanceof TrainableMemory;

        if (LAYER_0_TRAINABLE_MEMORY) {
            trainableMemoryLayer0 = (TrainableMemory) LAYER_0;
        }
        if (LAYER_0_TRAINABLE) {
            trainableLayer0 = (Trainable) LAYER_0;
        }
        if (LAYER_1_TRAINABLE_MEMORY) {
            trainableMemoryLayer1 = (TrainableMemory) LAYER_1;
        }
        if (LAYER_1_TRAINABLE) {
            trainableLayer1 = (Trainable) LAYER_1;
        }

        TRAINABLE = LAYER_0_TRAINABLE || LAYER_1_TRAINABLE;
        TRAINABLE_MEMORY = LAYER_0_TRAINABLE_MEMORY || LAYER_1_TRAINABLE_MEMORY;
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        layer0Input = inDimensions.clone();
        layer1Input = inDimensions.clone();
        layer0Input[layer0Input.length - 1] = LAYER_0_INPUT;
        layer1Input[layer1Input.length - 1] -= LAYER_0_INPUT;
        LAYER_0.setInDimensions(layer0Input);
        LAYER_1.setInDimensions(layer1Input);
        layer0Output = LAYER_0.getOutDimensions();
        layer1Output = LAYER_1.getOutDimensions();
        OUT_DIMENSIONS = LAYER_0.getOutDimensions().clone();
        int lastIndex = OUT_DIMENSIONS.length - 1;
        LAYER_0_OUTPUT = OUT_DIMENSIONS[lastIndex];
        LAYER_1_OUTPUT = layer1Output[lastIndex];
        OUT_DIMENSIONS[lastIndex] += LAYER_1_OUTPUT;
        length = 0;
        layer0WeightLength = 0;
        layer1WeightLength = 0;
        if (LAYER_0_TRAINABLE) {
            layer0WeightLength = trainableLayer0.weightShape().length;
            length += layer0WeightLength;
            if (LAYER_0_TRAINABLE_MEMORY) {
                layer0MemoryLength = trainableMemoryLayer0.getMemory().length;
                layer0Memory = LAYER_0_OUTPUT;
                layer0InputLength = trainableMemoryLayer0.getInput().size();
            }
        }
        if (LAYER_1_TRAINABLE) {
            layer1WeightLength = trainableLayer1.weightShape().length;
            length += layer1WeightLength;
            if (LAYER_1_TRAINABLE_MEMORY) {
                layer1MemoryLength = trainableMemoryLayer1.getMemory().length;
                layer1Memory = LAYER_1_OUTPUT;
                layer1InputLength = trainableMemoryLayer1.getInput().size();
            }
        }
        weightShape = new int[length][];
        weights = new Tensor[length];
        for (int i = 0; i < layer0WeightLength; i++) {
            weightShape[i] = trainableLayer0.weightShape()[i];
        }
        for (int i = 0; i < layer1WeightLength; i++) {
            weightShape[layer0WeightLength + i] = trainableLayer1.weightShape()[i];
        }
    }

    @Override
    public void learning(boolean on) {
        learningOn = on;
        LAYER_0.learning(on);
        LAYER_1.learning(on);
    }

    @Override
    public boolean learningOn() {
        return learningOn;
    }


    @Override
    public int[] getOutDimensions() {
        return OUT_DIMENSIONS;
    }

    @Override
    public Tensor feedForward(Tensor in) {
        Tensor[] inputs = Tensor.split(in, LAYER_0_INPUT);
        Tensor out0 = LAYER_0.feedForward(inputs[0]);
        Tensor out1 = LAYER_1.feedForward(inputs[1]);
        return Tensor.concat(out0, out1);
    }

    @Override
    public Tensor getDelta(Tensor outputError) {
        Tensor[] errors = Tensor.split(outputError, LAYER_0_OUTPUT);
        Tensor error0 = LAYER_0.getDelta(errors[0]);
        Tensor error1 = LAYER_1.getDelta(errors[1]);
        return Tensor.concat(error0, error1);
    }

    @Override
    public void initializeWeights() {
        LAYER_0.initializeWeights();
        LAYER_1.initializeWeights();
        for (int i = 0; i < layer0WeightLength; i++) {
            weights[i] = trainableLayer0.getWeights()[i];
        }
        for (int i = 0; i < layer1WeightLength; i++) {
            weights[layer0WeightLength + i] = trainableLayer1.getWeights()[i];
        }
    }

    @Override
    public int[][] weightShape() {
        return weightShape;
    }

    @Override
    public Tensor[] getGradients() {
        Tensor[] gradients = new Tensor[length];
        if (LAYER_0_TRAINABLE) {
            System.arraycopy(trainableLayer0.getGradients(), 0, gradients, 0, layer0WeightLength);
        }
        if (LAYER_1_TRAINABLE) {
            System.arraycopy(trainableLayer1.getGradients(), 0, gradients, layer0WeightLength, layer1WeightLength);
        }
        return gradients;
    }

    @Override
    public void setOutputError(Tensor outputError) {
        if (TRAINABLE) {
            Tensor[] outputErrors = Tensor.split(outputError, LAYER_0_OUTPUT);
            if (LAYER_0_TRAINABLE) {
                trainableLayer0.setOutputError(outputErrors[0]);
            }
            if (LAYER_1_TRAINABLE) {
                trainableLayer1.setOutputError(outputErrors[1]);
            }
        }
    }

    @Override
    public Tensor[] getWeights() {
        return weights;
    }

    @Override
    public void setMemory(Tensor[] memory) {
        if (LAYER_0_TRAINABLE_MEMORY) {
            Tensor[] layerMemory = new Tensor[layer0MemoryLength];
            System.arraycopy(memory, 0, layerMemory, 0, layer0MemoryLength);
            trainableMemoryLayer0.setMemory(layerMemory);
        }
        if (LAYER_1_TRAINABLE_MEMORY) {
            Tensor[] layerMemory = new Tensor[layer1MemoryLength];
            System.arraycopy(memory, layer0MemoryLength, layerMemory, 0, layer1MemoryLength);
            trainableMemoryLayer1.setMemory(layerMemory);
        }
    }

    @Override
    public void setNextMemory(Tensor[] memory) {
        if (LAYER_0_TRAINABLE_MEMORY) {
            Tensor[] layerMemory = new Tensor[layer0MemoryLength];
            System.arraycopy(memory, 0, layerMemory, 0, layer0MemoryLength);
            trainableMemoryLayer0.setNextMemory(layerMemory);
        }
        if (LAYER_1_TRAINABLE_MEMORY) {
            Tensor[] layerMemory = new Tensor[layer1MemoryLength];
            System.arraycopy(memory, layer0MemoryLength, layerMemory, 0, layer1MemoryLength);
            trainableMemoryLayer1.setNextMemory(layerMemory);
        }
    }

    @Override
    public void setInput(Tensor input) {
        if (TRAINABLE_MEMORY) {
            Tensor[] layerInputs = Tensor.split(input, LAYER_0_INPUT);
            if (LAYER_0_TRAINABLE_MEMORY) {
                trainableMemoryLayer0.setInput(layerInputs[0]);
            }
            if (LAYER_1_TRAINABLE_MEMORY) {
                trainableMemoryLayer1.setInput(layerInputs[1]);
            }
        }
    }

    @Override
    public void resetMemory() {
        if (LAYER_0_TRAINABLE_MEMORY) {
            trainableMemoryLayer0.resetMemory();
        }
        if (LAYER_1_TRAINABLE_MEMORY) {
            trainableMemoryLayer1.resetMemory();
        }
    }

    @Override
    public Tensor[] getMemory() {
        if (TRAINABLE_MEMORY) {
            Tensor[] memory0 = new Tensor[0];
            Tensor[] memory1 = new Tensor[0];
            if (LAYER_0_TRAINABLE_MEMORY) {
                memory0 = trainableMemoryLayer0.getMemory();
            }
            if (LAYER_1_TRAINABLE_MEMORY) {
                memory1 = trainableMemoryLayer1.getMemory();
            }
            Tensor[] memory = new Tensor[memory0.length + memory1.length];
            System.arraycopy(memory0, 0, memory, 0, memory0.length);
            System.arraycopy(memory1, 0, memory, memory0.length, memory1.length);
            return memory;
        }
        return null;
    }

    @Override
    public Tensor getInput() {
        if (TRAINABLE_MEMORY) {
            Tensor input0 = Tensor.zeros(layer0Input);
            Tensor input1 = Tensor.zeros(layer1Input);
            if (LAYER_0_TRAINABLE_MEMORY) {
                input0 = trainableMemoryLayer0.getInput();
            }
            if (LAYER_1_TRAINABLE_MEMORY) {
                input1 = trainableMemoryLayer1.getInput();
            }
            return Tensor.concat(input0, input1);
        }
        return null;
    }

    @Override
    public Tensor getMemoryDelta(Tensor delta) {
        if (TRAINABLE_MEMORY) {
            Tensor[] deltas = Tensor.split(delta, LAYER_0_OUTPUT);
            Tensor delta0 = Tensor.zeros(layer0Output);
            Tensor delta1 = Tensor.zeros(layer1Output);
            if (LAYER_0_TRAINABLE_MEMORY) {
                delta0 = trainableMemoryLayer0.getMemoryDelta(deltas[0]);
            }
            if (LAYER_1_TRAINABLE_MEMORY) {
                delta1 = trainableMemoryLayer1.getMemoryDelta(deltas[1]);
            }
            return Tensor.concat(delta0, delta1);
        }
        return null;
    }

    @Override
    public void calculateGradients(Tensor outputError) {
        if (TRAINABLE_MEMORY) {
            Tensor[] outputErrors = Tensor.split(outputError, LAYER_0_OUTPUT);
            if (LAYER_0_TRAINABLE_MEMORY) {
                trainableMemoryLayer0.calculateGradients(outputErrors[0]);
            }
            if (LAYER_1_TRAINABLE_MEMORY) {
                trainableMemoryLayer1.calculateGradients(outputErrors[1]);
            }
        }
    }
}
