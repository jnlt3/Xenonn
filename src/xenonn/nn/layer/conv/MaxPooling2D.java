package xenonn.nn.layer.conv;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;

public class MaxPooling2D extends Layer {


    private int IN_X;
    private int IN_Y;

    private final int POOL_SIZE;

    private int OUT_X;
    private int OUT_Y;
    private int OUT_IMAGES;

    private Tensor weights;

    public MaxPooling2D(int poolSize) {
        POOL_SIZE = poolSize;
    }

    public void initializeWeights() {

    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN_X = inDimensions[1];
        IN_Y = inDimensions[2];
        if (IN_X % POOL_SIZE != 0 || IN_Y % POOL_SIZE != 0) {
            throw new RuntimeException("Pool out of bounds");
        }
        OUT_X = IN_X / POOL_SIZE;
        OUT_Y = IN_Y / POOL_SIZE;
        OUT_IMAGES = inDimensions.length == 4 ? inDimensions[3] : 1;
        weights = Tensor.zeros(IN_X, IN_Y, POOL_SIZE, POOL_SIZE, OUT_IMAGES);
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, OUT_X, OUT_Y, OUT_IMAGES};
    }

    @Override
    public Tensor feedForward(Tensor in) {
        if (learningOn) {
            weights.zero();
        }
        Tensor out = Tensor.zeros(1, OUT_X, OUT_Y, OUT_IMAGES);
        for (int i = 0; i < OUT_IMAGES; i++) {
            for (int x = 0; x < IN_X; x += POOL_SIZE) {
                for (int y = 0; y < IN_Y; y += POOL_SIZE) {
                    double value = Double.NEGATIVE_INFINITY;
                    int pixelX = -1;
                    int pixelY = -1;
                    for (int kx = 0; kx < POOL_SIZE; kx++) {
                        for (int ky = 0; ky < POOL_SIZE; ky++) {
                            int imageX = x + kx;
                            int imageY = y + ky;
                            double pixelValue = in.get(0, imageX, imageY, i);
                            if (pixelValue > value) {
                                value = pixelValue;
                                pixelX = kx;
                                pixelY = ky;
                            }
                        }
                    }
                    if (Double.isInfinite(value)) {
                        System.out.println(in);
                    }
                    if (learningOn) {
                        weights.set(new int[]{x + pixelX, y + pixelY, pixelX, pixelY, i}, 1);
                    }
                    //System.out.println(Arrays.toString(new int[]{x + pixelX, y + pixelY, pixelX, pixelY}));
                    out.set(new int[]{0, x / POOL_SIZE, y / POOL_SIZE, i}, value);
                }
            }
        }
        return out;
    }



    @Override
    public Tensor getDelta(Tensor outputError) {
        Tensor delta = Tensor.zeros(1, IN_X, IN_Y, OUT_IMAGES);
        for (int i = 0; i < OUT_IMAGES; i++) {
            for (int x = 0; x < OUT_X; x++) {
                for (int y = 0; y < OUT_Y; y++) {
                    for (int kx = 0; kx < POOL_SIZE; kx++) {
                        for (int ky = 0; ky < POOL_SIZE; ky++) {
                            int imageX = x * POOL_SIZE + kx;
                            int imageY = y * POOL_SIZE + ky;
                            int[] indices = new int[]{0, x / POOL_SIZE, y / POOL_SIZE, i};
                            delta.set(indices, delta.get(indices) + outputError.get(0, x, y, i) * weights.get(imageX, imageY, kx, ky, i));
                        }
                    }
                }
            }
        }
        return delta;
    }
}
