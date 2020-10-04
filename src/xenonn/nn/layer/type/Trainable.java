package xenonn.nn.layer.type;

import xenonn.math.Tensor;

public interface Trainable {

    int[][] weightShape();

    Tensor[] getGradients();

    void setOutputError(Tensor outputError);

    Tensor[] getWeights();

}
