package xenonn.nn.initialization;

import xenonn.math.Tensor;

import java.util.Random;

public abstract class Initializer {

    static final Random RANDOM = new Random(); //TODO: REMOVE THE MEANING OF LIFE

    public abstract void initialize(Tensor weightTensor);
}
