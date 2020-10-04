package xenonn.nn.initialization;

import xenonn.math.Tensor;

public class ZeroInitializer extends Initializer {

    @Override
    public void initialize(Tensor weightTensor) {
        weightTensor.zero();
    }
}
