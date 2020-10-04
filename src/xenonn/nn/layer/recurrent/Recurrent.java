package xenonn.nn.layer.recurrent;

import xenonn.math.Tensor;
import xenonn.nn.activation.ActivationFunction;
import xenonn.nn.initialization.Initializer;
import xenonn.nn.initialization.NullInitializer;
import xenonn.nn.initialization.XavierInitializerNormal;
import xenonn.nn.layer.Layer;
import xenonn.nn.layer.type.Trainable;
import xenonn.nn.layer.type.TrainableMemory;

public class Recurrent extends Layer implements Trainable, TrainableMemory {

    private final ActivationFunction ACTIVATION_FUNCTION;
    private final Initializer INITIALIZER;

    private final int MEMORY_CELLS;
    private int IN;

    private final Tensor[] WEIGHTS;
    private int[][] weightShape;


    private Tensor[] memoryGradients = new Tensor[3];

    private Tensor lastInput;
    private transient Tensor lastMemory;
    private transient Tensor activatedMemory;

    private Tensor learningLastInput;
    private transient Tensor learningLastMemory;
    private transient Tensor learningActivatedMemory;
    private transient Tensor learningNextMemoryDerivative;
    private transient Tensor learningNextActivatedMemory;

    public Recurrent(int memoryCells, ActivationFunction activationFunction) {
        this(memoryCells, activationFunction, new XavierInitializerNormal());
    }

    public Recurrent(int memoryCells, ActivationFunction activationFunction, Tensor[] weights) {
        this(memoryCells, activationFunction, new NullInitializer());
        System.arraycopy(weights, 0, WEIGHTS, 0, 3);
    }

    public Recurrent(int memoryCells, ActivationFunction activationFunction, Initializer initializer) {
        WEIGHTS = new Tensor[3]; //Wax, Waa, Ba
        MEMORY_CELLS = memoryCells;
        ACTIVATION_FUNCTION = activationFunction;
        INITIALIZER = initializer;
    }

    public void initializeWeights() {
        INITIALIZER.initialize(WEIGHTS[0]);
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN = inDimensions[1];
        weightShape = new int[][]{{IN, MEMORY_CELLS}, {MEMORY_CELLS, MEMORY_CELLS}, {1, MEMORY_CELLS}};
        for (int i = 0; i < weightShape.length; i++) {
            if (WEIGHTS[i] == null) {
                WEIGHTS[i] = Tensor.zeros(weightShape[i]);
            }
            memoryGradients[i] = Tensor.zeros(weightShape[i]);
        }
        lastInput = Tensor.zeros(inDimensions);
        learningLastInput = Tensor.zeros(inDimensions);
        learningActivatedMemory = Tensor.zeros(weightShape[2]);
        lastMemory = Tensor.zeros(weightShape[2]);
        learningLastMemory = Tensor.zeros(weightShape[2]);
        activatedMemory = Tensor.zeros(weightShape[2]);
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, MEMORY_CELLS};
    }

    @Override
    public Tensor feedForward(Tensor in) {
        if (learningOn) {
            learningLastInput = in.copy();
            //learningLastMemory = Tensor.add(Tensor.add(Tensor.dotProduct(in, WEIGHTS[0]), Tensor.dotProduct(learningActivatedMemory, WEIGHTS[1])), WEIGHTS[2]);
            learningLastMemory = Tensor.dotProduct(in, WEIGHTS[0]).add(Tensor.dotProduct(learningActivatedMemory, WEIGHTS[1])).add(WEIGHTS[2]);
            learningActivatedMemory = ACTIVATION_FUNCTION.value(learningLastMemory);
            return learningActivatedMemory.copy();
        }
        lastInput = in.copy();
        //lastMemory = Tensor.add(Tensor.add(Tensor.dotProduct(in, WEIGHTS[0]), Tensor.dotProduct(activatedMemory, WEIGHTS[1])), WEIGHTS[2]);
        learningLastMemory = Tensor.dotProduct(in, WEIGHTS[0]).add(Tensor.dotProduct(activatedMemory, WEIGHTS[1])).add(WEIGHTS[2]);
        activatedMemory = ACTIVATION_FUNCTION.value(lastMemory);
        return activatedMemory.copy();
    }

    @Override
    public Tensor[] getWeights() {
        return WEIGHTS;
    }


    @Override
    public Tensor getDelta(Tensor outputError) {
        return Tensor.dotProduct(outputError, Tensor.transpose(WEIGHTS[0]));
    }

    @Override
    public int[][] weightShape() {
        return weightShape;
    }

    @Override
    public Tensor[] getGradients() {
        return memoryGradients;
    }

    @Override
    public void setOutputError(Tensor outputError) {
    }

    @Override
    public void setMemory(Tensor[] memory) {
        if (learningOn) {
            learningLastMemory = memory[0].copy();
            learningActivatedMemory = memory[1].copy();
            return;
        }
        lastMemory = memory[0].copy();
        activatedMemory = memory[1].copy();
    }

    @Override
    public void setNextMemory(Tensor[] memory) {
        if (learningOn) {
            learningNextMemoryDerivative = ACTIVATION_FUNCTION.derivative(memory[0].copy());
            learningNextActivatedMemory = memory[1].copy();
        }
    }

    @Override
    public void setInput(Tensor input) {
        if (learningOn) {
            learningLastInput = input.copy();
            return;
        }
        lastInput = input.copy();
    }

    @Override
    public void resetMemory() {
        if (learningOn) {
            learningLastMemory.zero();
            learningActivatedMemory.zero();
            return;
        }
        lastMemory.zero();
        activatedMemory.zero();
    }

    @Override
    public Tensor[] getMemory() {
        if (learningOn) {
            return new Tensor[]{learningLastMemory.copy(), learningActivatedMemory.copy()};
        }
        return new Tensor[]{lastMemory.copy(), activatedMemory.copy()};
    }

    @Override
    public Tensor getInput() {
        if (learningOn) {
            return learningLastInput.copy();
        }
        return lastInput.copy();
    }

    @Override
    public Tensor getMemoryDelta(Tensor delta) {
        return Tensor.dotProduct(Tensor.multiply(delta, learningNextMemoryDerivative), Tensor.transpose(WEIGHTS[1]));
    }

    @Override
    public void calculateGradients(Tensor outputError) {
        memoryGradients[0].add(Tensor.dotProduct(Tensor.transpose(learningLastInput), outputError));
        memoryGradients[1].add(Tensor.dotProduct(Tensor.transpose(learningNextActivatedMemory), outputError));
        memoryGradients[2].add(outputError);
    }
}
