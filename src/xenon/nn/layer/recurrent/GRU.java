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

import java.util.Arrays;

public class GRU extends Layer implements Trainable, TrainableMemory {

    private final ActivationFunction SIGMOID = new Sigmoid();
    private final ActivationFunction TANH = new Tanh();
    private final Initializer INITIALIZER;

    private int IN;
    private final int MEMORY_CELLS;

    private final Tensor[] WEIGHTS;
    private int[][] weightShape = new int[9][];


    private Tensor[] gradients = new Tensor[9];

    private transient Tensor lastInput;
    private transient Tensor candidateOutput;
    private transient Tensor activatedCandidateOutput;
    private Tensor output;
    private transient Tensor[] gates = new Tensor[2];

    private transient Tensor learningLastInput;
    private transient Tensor learningCandidateOutput;
    private transient Tensor learningActivatedCandidateOutput;
    private Tensor learningOutput;
    private transient Tensor[] learningGates = new Tensor[2];
    private transient Tensor[] learningGateDerivatives = new Tensor[2];
    private transient Tensor learningDerivativeCandidateOutput;

    private transient Tensor nextLearningCandidateOutput;
    private transient Tensor nextLearningActivatedCandidateOutput;

    public GRU(int memoryCells) {
        this(memoryCells, new XavierInitializerNormal());
    }

    public GRU(int memoryCells, Tensor[] weights) {
        this(memoryCells, new NullInitializer());
        System.arraycopy(weights, 0, WEIGHTS, 0, 9);
    }

    public GRU(int memoryCells, Initializer initializer) {
        WEIGHTS = new Tensor[9]; //update reset candidate
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
        for (int i = 0; i < 9; i += 3) {
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
        output = Tensor.zeros(1, MEMORY_CELLS);
        candidateOutput = Tensor.zeros(1, MEMORY_CELLS);
        activatedCandidateOutput = Tensor.zeros(1, MEMORY_CELLS);

        learningLastInput = Tensor.zeros(inDimensions);
        learningOutput = Tensor.zeros(1, MEMORY_CELLS);
        learningCandidateOutput = Tensor.zeros(1, MEMORY_CELLS);
        learningActivatedCandidateOutput = Tensor.zeros(1, MEMORY_CELLS);
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, MEMORY_CELLS};
    }

    @Override
    public Tensor feedForward(Tensor in) {
        if (learningOn) {
            learningLastInput = in.copy();
            Tensor[] activatedGates = new Tensor[2]; //update reset
            for (int i = 0; i < activatedGates.length; i++) {
                int index = i * 3;
                learningGates[i] = Tensor.add(Tensor.add(Tensor.dotProduct(in, WEIGHTS[index]), Tensor.dotProduct(learningOutput, WEIGHTS[index + 1])), WEIGHTS[index + 2]);
                activatedGates[i] = SIGMOID.value(learningGates[i]);
            }
            learningCandidateOutput = Tensor.add(Tensor.dotProduct(in, WEIGHTS[6]), Tensor.dotProduct(Tensor.multiply(activatedGates[0], learningOutput), WEIGHTS[7]));
            learningActivatedCandidateOutput = TANH.value(learningCandidateOutput);
            learningOutput = Tensor.add(Tensor.multiply(Tensor.subtract(1, activatedGates[1]), learningOutput), Tensor.multiply(activatedGates[1], learningActivatedCandidateOutput));
            return learningOutput.copy();
        }
        lastInput = in.copy();
        Tensor[] activatedGates = new Tensor[2]; //update reset
        for (int i = 0; i < activatedGates.length; i++) {
            int index = i * 3;
            gates[i] = Tensor.add(Tensor.add(Tensor.dotProduct(in, WEIGHTS[index]), Tensor.dotProduct(output, WEIGHTS[index + 1])), WEIGHTS[index + 2]);
            activatedGates[i] = SIGMOID.value(gates[i]);
        }
        candidateOutput = Tensor.add(Tensor.dotProduct(in, WEIGHTS[6]), Tensor.dotProduct(Tensor.multiply(activatedGates[0], output), WEIGHTS[7]));
        activatedCandidateOutput = TANH.value(candidateOutput);
        output = Tensor.add(Tensor.multiply(Tensor.subtract(1, activatedGates[1]), output), Tensor.multiply(activatedGates[1], activatedCandidateOutput));
        return output.copy();
    }

    @Override
    public Tensor[] getWeights() {
        return WEIGHTS;
    }


    @Override
    public Tensor getDelta(Tensor outputError) {
        learningDerivativeCandidateOutput = SIGMOID.value(learningGates[0]).multiply(TANH.derivative(nextLearningCandidateOutput)).multiply(outputError);
        learningGateDerivatives[0] = Tensor.subtract(nextLearningActivatedCandidateOutput, learningOutput).multiply(outputError).multiply(SIGMOID.derivative(learningGates[0]));
        learningGateDerivatives[1] = Tensor.dotProduct(Tensor.multiply(learningDerivativeCandidateOutput, learningOutput), Tensor.transpose(WEIGHTS[7])).multiply(SIGMOID.derivative(learningGates[1]));
        Tensor delta = Tensor.zeros(1, IN);
        for (int i = 0; i < 2; i++) {
            int index = i * 3;
            delta.add(Tensor.dotProduct(learningGateDerivatives[i], Tensor.transpose(WEIGHTS[index])));
        }
        delta.add(Tensor.dotProduct(learningDerivativeCandidateOutput, Tensor.transpose(WEIGHTS[6])));
        return delta;
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
            learningOutput = memory[0].copy();
            learningCandidateOutput = memory[1].copy();
            learningActivatedCandidateOutput = memory[2].copy();
            return;
        }
        output = memory[0].copy();
        candidateOutput = memory[1].copy();
        activatedCandidateOutput = memory[2].copy();
    }

    @Override
    public void setNextMemory(Tensor[] memory) {
        nextLearningCandidateOutput = memory[1].copy();
        nextLearningActivatedCandidateOutput = memory[2].copy();
        for (int i = 3; i < memory.length; i++) {
            learningGates[i - 3] = memory[i].copy();
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
            learningOutput.zero();
            return;
        }
        output.zero();
    }

    @Override
    public Tensor[] getMemory() {
        if (learningOn) {
            Tensor[] memory = new Tensor[3 + gates.length];
            memory[0] = learningOutput.copy();
            memory[1] = learningCandidateOutput.copy();
            memory[2] = learningActivatedCandidateOutput.copy();
            for (int i = 3; i < memory.length; i++) {
                memory[i] = learningGates[i - 3].copy();
            }
            return memory;
        }
        Tensor[] memory = new Tensor[3 + gates.length];
        memory[0] = output.copy();
        memory[1] = candidateOutput.copy();
        memory[2] = activatedCandidateOutput.copy();
        for (int i = 3; i < memory.length; i++) {
            memory[i] = gates[i - 3].copy();
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
        Tensor[] gateErrors = new Tensor[2];
        for (int i = 0; i < gateErrors.length; i++) {
            gateErrors[i] = Tensor.dotProduct(learningGateDerivatives[i], Tensor.transpose(WEIGHTS[i * 3 + 1]));
        }
        return gateErrors[0].add(gateErrors[1]).add(learningDerivativeCandidateOutput);
    }

    @Override
    public void calculateGradients(Tensor outputError) {
        learningDerivativeCandidateOutput = SIGMOID.value(learningGates[0]).multiply(TANH.derivative(nextLearningCandidateOutput)).multiply(outputError);
        learningGateDerivatives[0] = Tensor.subtract(nextLearningActivatedCandidateOutput, learningOutput).multiply(outputError).multiply(SIGMOID.derivative(learningGates[0]));
        learningGateDerivatives[1] = Tensor.dotProduct(Tensor.multiply(learningDerivativeCandidateOutput, learningOutput), Tensor.transpose(WEIGHTS[7])).multiply(SIGMOID.derivative(learningGates[1]));
        for (int i = 0; i < 2; i++) {
            int index = i * 3;
            gradients[index].add(Tensor.dotProduct(Tensor.transpose(learningLastInput), learningGateDerivatives[i]));
            gradients[index + 1].add(Tensor.dotProduct(Tensor.transpose(learningOutput), learningGateDerivatives[i]));
            gradients[index + 2].add(learningGateDerivatives[i]);
        }
        gradients[6].add(Tensor.dotProduct(Tensor.transpose(learningLastInput), learningDerivativeCandidateOutput));
        gradients[7].add(Tensor.dotProduct(Tensor.transpose(Tensor.multiply(SIGMOID.value(learningGates[1]), learningOutput)), learningDerivativeCandidateOutput));
        gradients[8].add(learningDerivativeCandidateOutput);
    }
}


