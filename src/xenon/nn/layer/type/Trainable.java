package xenon.nn.layer.type;

import xenon.math.Tensor;

public interface Trainable {

    int[][] weightShape();

    Tensor[] getGradients();

    void setOutputError(Tensor outputError);

    Tensor[] getWeights();

}
