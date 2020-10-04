package xenon.nn.layer.recurrent;

import xenon.math.Tensor;
import xenon.nn.activation.ActivationFunction;
import xenon.nn.activation.Sigmoid;
import xenon.nn.activation.Tanh;
import xenon.nn.initialization.Initializer;
import xenon.nn.initialization.NullInitializer;
import xenon.nn.initialization.XavierInitializerNormal;
import xenon.nn.layer.Layer;
import xenon.nn.layer.type.Trainable;
import xenon.nn.layer.type.TrainableMemory;

public class LSTM extends Layer implements Trainable, TrainableMemory {

    private final ActivationFunction SIGMOID = new Sigmoid();
    private final ActivationFunction TANH = new Tanh();
    private final Initializer INITIALIZER;

    private int IN;
    private final int MEMORY_CELLS;

    private final Tensor[] WEIGHTS;
    private int[][] weightShape = new int[12][];


    private Tensor[] gradients = new Tensor[12];

    private transient Tensor lastInput;
    private Tensor memoryCell;
    private transient Tensor activatedMemoryCell;
    private transient Tensor candidateMemoryCell;
    private transient Tensor activatedCandidateMemoryCell;
    private Tensor outputCell;
    private transient Tensor[] gates = new Tensor[9];

    private transient Tensor learningLastInput;
    private Tensor learningMemoryCell;
    private transient Tensor learningActivatedMemoryCell;
    private transient Tensor learningCandidateMemoryCell;
    private transient Tensor learningActivatedCandidateMemoryCell;
    private Tensor learningOutputCell;

    private transient Tensor learningNextMemoryCell;
    private transient Tensor learningNextActivatedMemoryCell;
    private transient Tensor learningNextCandidateMemoryCell;
    private transient Tensor learningNextActivatedCandidateMemoryCell;
    private transient Tensor learningNextOutputCell;
    private transient Tensor[] learningGates = new Tensor[9];
    private transient Tensor[] learningGateDerivatives = new Tensor[9];
    private transient Tensor candidateDerivative;

    public LSTM(int memoryCells) {
        this(memoryCells, new XavierInitializerNormal());
    }

    public LSTM(int memoryCells, Tensor[] weights) {
        this(memoryCells, new NullInitializer());
        System.arraycopy(weights, 0, WEIGHTS, 0, 12);
    }

    public LSTM(int memoryCells, Initializer initializer) {
        WEIGHTS = new Tensor[12]; //forget input output candidate
        MEMORY_CELLS = memoryCells;
        INITIALIZER = initializer;
    }

    public void initializeWeights() {
        for (int i = 0; i < WEIGHTS.length; i += 3) {
            INITIALIZER.initialize(WEIGHTS[i]);
            INITIALIZER.initialize(WEIGHTS[i + 2]);
        }
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN = inDimensions[1];
        int[][] repeatingWeightShape = new int[][]{{IN, MEMORY_CELLS}, {MEMORY_CELLS, MEMORY_CELLS}, {1, MEMORY_CELLS}};
        for (int i = 0; i < 12; i += 3) {
            System.arraycopy(repeatingWeightShape, 0, weightShape, i, 3);
        }
        for (int i = 0; i < weightShape.length; i++) {
            if (WEIGHTS[i] == null) {
                WEIGHTS[i] = Tensor.zeros(weightShape[i]);
            }
            gradients[i] = Tensor.zeros(weightShape[i]);
        }
        for (int i = 0; i < learningGates.length; i++) {
            gates[i] = Tensor.zeros(weightShape[i]);
            learningGates[i] = Tensor.zeros(weightShape[i]);
        }
        lastInput = Tensor.zeros(inDimensions);
        memoryCell = Tensor.zeros(1, MEMORY_CELLS);
        candidateMemoryCell = Tensor.zeros(1, MEMORY_CELLS);
        activatedMemoryCell = Tensor.zeros(1, MEMORY_CELLS);
        activatedCandidateMemoryCell = Tensor.zeros(1, MEMORY_CELLS);
        outputCell = Tensor.zeros(1, MEMORY_CELLS);
        learningOutputCell = Tensor.zeros(1, MEMORY_CELLS);
        learningLastInput = Tensor.zeros(inDimensions);
        learningMemoryCell = Tensor.zeros(1, MEMORY_CELLS);
        learningCandidateMemoryCell = Tensor.zeros(1, MEMORY_CELLS);
        learningActivatedCandidateMemoryCell = Tensor.zeros(1, MEMORY_CELLS);
        learningActivatedMemoryCell = Tensor.zeros(1, MEMORY_CELLS);
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, MEMORY_CELLS};
    }

    @Override
    public Tensor feedForward(Tensor in) {
        if (learningOn) {
            learningLastInput = in.copy();
            Tensor[] activatedGates = new Tensor[3]; //forget, input, output
            for (int i = 0; i < activatedGates.length; i++) {
                int index = i * 3;
                learningGates[i] = Tensor.dotProduct(in, WEIGHTS[index]).add(Tensor.dotProduct(learningOutputCell, WEIGHTS[index + 1])).add(WEIGHTS[index + 2]);
                activatedGates[i] = SIGMOID.value(learningGates[i]);
            }
            learningCandidateMemoryCell = Tensor.dotProduct(in, WEIGHTS[9]).add(Tensor.dotProduct(learningOutputCell, WEIGHTS[10])).add(WEIGHTS[11]);
            learningActivatedCandidateMemoryCell = TANH.value(learningCandidateMemoryCell);
            learningMemoryCell = Tensor.multiply(activatedGates[0], learningMemoryCell).add(Tensor.multiply(activatedGates[1], learningCandidateMemoryCell));
            learningActivatedMemoryCell = TANH.value(learningMemoryCell);
            learningOutputCell = Tensor.multiply(activatedGates[2], learningActivatedMemoryCell);
            return learningOutputCell.copy();
        }
        lastInput = in.copy();
        Tensor[] activatedGates = new Tensor[3]; //forget, input, output
        for (int i = 0; i < activatedGates.length; i++) {
            int index = i * 3;
            gates[i] = Tensor.dotProduct(in, WEIGHTS[index]).add(Tensor.dotProduct(outputCell, WEIGHTS[index + 1])).add(WEIGHTS[index + 2]);
            activatedGates[i] = SIGMOID.value(gates[i]);
        }
        candidateMemoryCell = Tensor.dotProduct(in, WEIGHTS[9]).add(Tensor.dotProduct(outputCell, WEIGHTS[10])).add(WEIGHTS[11]);
        activatedCandidateMemoryCell = TANH.value(candidateMemoryCell);
        memoryCell = Tensor.multiply(activatedGates[0], memoryCell).add(Tensor.multiply(activatedGates[1], candidateMemoryCell));
        activatedMemoryCell = TANH.value(memoryCell);
        outputCell = Tensor.multiply(activatedGates[2], activatedMemoryCell);
        return outputCell.copy();
    }

    @Override
    public Tensor[] getWeights() {
        return WEIGHTS;
    }


    @Override
    public Tensor getDelta(Tensor outputError) {
        Tensor derivativeCell = Tensor.multiply(outputError, TANH.derivative(learningNextMemoryCell));
        Tensor derivativeActivatedOutputGate = Tensor.multiply(outputError, learningNextActivatedMemoryCell);
        learningGateDerivatives[0] = Tensor.multiply(Tensor.multiply(derivativeCell, learningMemoryCell), SIGMOID.derivative(learningGates[0]));
        learningGateDerivatives[1] = Tensor.multiply(Tensor.multiply(derivativeCell, learningNextActivatedCandidateMemoryCell), SIGMOID.derivative(learningGates[1]));
        learningGateDerivatives[2] = Tensor.multiply(derivativeActivatedOutputGate, SIGMOID.derivative(learningGates[2]));
        Tensor[] gateErrors = new Tensor[3];
        for (int i = 0; i < gateErrors.length; i++) {
            gateErrors[i] = Tensor.dotProduct(learningGateDerivatives[i], Tensor.transpose(WEIGHTS[i * 3]));
        }
        candidateDerivative = Tensor.multiply(Tensor.multiply(derivativeCell, SIGMOID.value(learningGates[1])), TANH.derivative(learningNextCandidateMemoryCell));
        Tensor candidateError = Tensor.dotProduct(candidateDerivative, Tensor.transpose(WEIGHTS[9]));
        return Tensor.add(Tensor.add(gateErrors[0], gateErrors[1]), Tensor.add(gateErrors[2], candidateError));
    }

    @Override
    public int[][] weightShape() {
        return weightShape;
    }

    @Override
    public Tensor[] getGradients() {
        return gradients;
    }

    @Override
    public void setOutputError(Tensor outputError) {

    }

    @Override
    public void setMemory(Tensor[] memory) {
        if (learningOn) {
            learningMemoryCell = memory[0].copy();
            learningActivatedMemoryCell = memory[1].copy();
            learningCandidateMemoryCell = memory[2].copy();
            learningOutputCell = memory[3].copy();
            learningActivatedCandidateMemoryCell = memory[4].copy();
            return;
        }
        memoryCell = memory[0].copy();
        activatedMemoryCell = memory[1].copy();
        candidateMemoryCell = memory[2].copy();
        outputCell = memory[3].copy();
        activatedCandidateMemoryCell = memory[4].copy();
    }

    @Override
    public void setNextMemory(Tensor[] memory) {
        learningNextMemoryCell = memory[0].copy();
        learningNextActivatedMemoryCell = memory[1].copy();
        learningNextCandidateMemoryCell = memory[2].copy();
        learningNextOutputCell = memory[3].copy();
        learningNextActivatedCandidateMemoryCell = memory[4].copy();
        for (int i = 5; i < memory.length; i++) {
            learningGates[i - 5] = memory[i].copy();
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
            learningMemoryCell.zero();
            learningOutputCell.zero();
            return;
        }
        memoryCell.zero();
        outputCell.zero();
    }

    @Override
    public Tensor[] getMemory() {
        if (learningOn) {
            Tensor[] memory = new Tensor[5 + gates.length];
            memory[0] = learningMemoryCell.copy();
            memory[1] = learningActivatedMemoryCell.copy();
            memory[2] = learningCandidateMemoryCell.copy();
            memory[3] = learningOutputCell.copy();
            memory[4] = learningActivatedCandidateMemoryCell.copy();
            for (int i = 5; i < memory.length; i++) {
                memory[i] = learningGates[i - 5].copy();
            }
            return memory;
        }
        Tensor[] memory = new Tensor[5 + gates.length];
        memory[0] = memoryCell.copy();
        memory[1] = activatedMemoryCell.copy();
        memory[2] = candidateMemoryCell.copy();
        memory[3] = outputCell.copy();
        memory[4] = activatedCandidateMemoryCell.copy();
        for (int i = 5; i < memory.length; i++) {
            memory[i] = gates[i - 5].copy();
        }
        return memory;
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
        Tensor[] gateErrors = new Tensor[3];
        for (int i = 0; i < gateErrors.length; i++) {
            gateErrors[i] = Tensor.dotProduct(learningGateDerivatives[i], Tensor.transpose(WEIGHTS[i * 3 + 1]));
        }
        Tensor candidateError = Tensor.dotProduct(candidateDerivative, Tensor.transpose(WEIGHTS[10]));
        return candidateError.add(gateErrors[0]).add(gateErrors[1]).add(gateErrors[2]);
    }

    @Override
    public void calculateGradients(Tensor outputError) {
        Tensor derivativeCell = Tensor.multiply(outputError, TANH.derivative(learningNextMemoryCell));
        Tensor derivativeActivatedOutputGate = Tensor.multiply(outputError, learningNextActivatedMemoryCell);
        learningGateDerivatives[0] = Tensor.multiply(derivativeCell, learningMemoryCell).multiply(SIGMOID.derivative(learningGates[0]));
        learningGateDerivatives[1] = Tensor.multiply(derivativeCell, learningNextActivatedCandidateMemoryCell).multiply(SIGMOID.derivative(learningGates[1]));
        learningGateDerivatives[2] = Tensor.multiply(derivativeActivatedOutputGate, SIGMOID.derivative(learningGates[2]));
        for (int i = 0; i < 3; i++) {
            int index = i * 3;
            gradients[index].add(Tensor.dotProduct(Tensor.transpose(learningLastInput), learningGateDerivatives[i]));
            gradients[index + 1].add(Tensor.dotProduct(Tensor.transpose(learningOutputCell), learningGateDerivatives[i]));
            gradients[index + 2].add(learningGateDerivatives[i]);
        }
        //candidateDerivative = Tensor.multiply(Tensor.multiply(derivativeCell, SIGMOID.value(learningGates[1])), TANH.derivative(learningNextCandidateMemoryCell));
        candidateDerivative = Tensor.multiply(derivativeCell, SIGMOID.value(learningGates[1])).multiply(TANH.derivative(learningNextCandidateMemoryCell));
        gradients[9].add(Tensor.dotProduct(Tensor.transpose(learningLastInput), candidateDerivative));
        gradients[10].add(Tensor.dotProduct(Tensor.transpose(learningOutputCell), candidateDerivative));
        gradients[11].add(candidateDerivative);
    }
}


