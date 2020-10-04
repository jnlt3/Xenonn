package xenon.nn.initialization;

import xenon.math.Tensor;

public class ZeroInitializer extends Initializer {

    @Override
    public void initialize(Tensor weightTensor) {
        weightTensor.zero();
    }
}
