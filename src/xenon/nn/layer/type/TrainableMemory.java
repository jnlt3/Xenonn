package xenon.nn.layer.type;

import xenon.math.Tensor;

public interface TrainableMemory extends Trainable {

    void setMemory(Tensor[] memory);

    void setNextMemory(Tensor[] memory);

    void setInput(Tensor input);

    void resetMemory();

    Tensor[] getMemory();

    Tensor getInput();

    Tensor getMemoryDelta(Tensor delta);

    void calculateGradients(Tensor outputError);
}
