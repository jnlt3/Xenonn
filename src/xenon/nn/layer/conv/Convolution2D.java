package xenon.nn.layer.conv;

import xenon.math.Tensor;
import xenon.nn.initialization.HeInitializerNormal;
import xenon.nn.initialization.Initializer;
import xenon.nn.layer.Layer;
import xenon.nn.layer.type.Trainable;

import java.util.Arrays;

public class Convolution2D extends Layer implements Trainable {


    private int IN_X;
    private int IN_Y;
    private int IN_IMAGES;

    private int NUM_FILTERS;
    private final int KERNEL_SIZE;

    private final boolean ZERO_PADDING;

    private int OUT_X;
    private int OUT_Y;

    private final Initializer INITIALIZER;


    private Tensor[] kernel;

    private Tensor outputError;
    private Tensor inputs;
    private Tensor delta;

    private Tensor[] gradients;

    private int[][] weightShape;

    public Convolution2D(int numberOfFilters, int kernelSize, boolean zeroPadding) {
        this(numberOfFilters, kernelSize, zeroPadding, new HeInitializerNormal());
    }

    public Convolution2D(Tensor[] weights, int numberOfFilters, int kernelSize, boolean zeroPadding, Initializer initializer) {
        NUM_FILTERS = numberOfFilters;
        KERNEL_SIZE = kernelSize;
        ZERO_PADDING = zeroPadding;

        INITIALIZER = initializer;
    }

    public Convolution2D(int numberOfFilters, int kernelSize, boolean zeroPadding, Initializer initializer) {
        NUM_FILTERS = numberOfFilters;
        KERNEL_SIZE = kernelSize;
        ZERO_PADDING = zeroPadding;

        INITIALIZER = initializer;
    }

    @Override
    public void setInDimensions(int[] inDimensions) {
        IN_X = inDimensions[1];
        IN_Y = inDimensions[2];
        IN_IMAGES = inDimensions.length == 4 ? inDimensions[3] : 1;
        if (ZERO_PADDING) {
            OUT_X = IN_X;
            OUT_Y = IN_Y;
        } else {
            OUT_X = IN_X - KERNEL_SIZE + 1;
            OUT_Y = IN_Y - KERNEL_SIZE + 1;
        }
        kernel = new Tensor[NUM_FILTERS * 2];
        gradients = new Tensor[NUM_FILTERS * 2];
        weightShape = new int[NUM_FILTERS * 2][];
        int[] kernelShape = new int[]{KERNEL_SIZE, KERNEL_SIZE, IN_IMAGES};
        int[] biasShape = new int[]{IN_IMAGES};
        for (int i = 0; i < kernel.length; i += 2) {
            kernel[i] = Tensor.zeros(KERNEL_SIZE, KERNEL_SIZE, IN_IMAGES);
            kernel[i + 1] = Tensor.zeros(IN_IMAGES);
            weightShape[i] = kernelShape;
            weightShape[i + 1] = biasShape;
        }
    }

    @Override
    public int[] getOutDimensions() {
        return new int[]{1, OUT_X, OUT_Y, NUM_FILTERS};
    }

    @Override
    public void initializeWeights() {
        if (INITIALIZER != null) {
            for (int i = 0; i < NUM_FILTERS * 2; i += 2) {
                INITIALIZER.initialize(kernel[i]);
            }
        }
    }

    @Override
    public Tensor feedForward(Tensor in) {
        in.reshape(1, IN_X, IN_Y, IN_IMAGES);
        if (learningOn) {
            inputs = in.copy();
        }
        Tensor out = Tensor.zeros(1, OUT_X, OUT_Y, NUM_FILTERS);
        for (int inF = 0; inF < IN_IMAGES; inF++) {
            for (int outF = 0; outF < NUM_FILTERS; outF++) {
                int kernelIndex = outF * 2;
                for (int x = 0; x < OUT_X; x++) {
                    for (int y = 0; y < OUT_Y; y++) {
                        double value = 0;
                        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                                int imageX = x + kx;
                                int imageY = y + ky;
                                if (!outOfBoundsForIn(imageX, imageY)) {
                                    value += in.get(0, imageX, imageY) * kernel[kernelIndex].get(kx, ky, inF);
                                }
                            }
                        }
                        int[] indices = new int[]{0, x, y, outF};
                        out.set(indices, out.get(indices) + value + kernel[kernelIndex + 1].get(inF));
                    }
                }
            }
        }

        return out;
    }

    @Override
    public Tensor[] getWeights() {
        return kernel;
    }


    @Override
    public int[][] weightShape() {
        return weightShape;
    }

    @Override
    public void setOutputError(Tensor outputError) {
        this.outputError = outputError.copy();
    }

    @Override
    public Tensor[] getGradients() {
        for (int i = 0; i < gradients.length; i += 2) {
            gradients[i] = Tensor.zeros(KERNEL_SIZE, KERNEL_SIZE, IN_IMAGES);
            gradients[i + 1] = Tensor.zeros(IN_IMAGES);
        }

        for (int inF = 0; inF < IN_IMAGES; inF++) {
            for (int outF = 0; outF < NUM_FILTERS; outF++) {
                int kernelIndex = outF * 2;
                int[] biasIndex = new int[]{inF};
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                        int[] kernelIndices = {kx, ky, inF};
                        for (int x = 0; x < OUT_X; x++) {
                            for (int y = 0; y < OUT_Y; y++) {
                                int[] inImageIndex = {0, x + kx, y + ky, inF};
                                int[] outImageIndex = {0, x, y, outF};
                                if (!outOfBoundsForIn(inImageIndex[1], inImageIndex[2])) {
                                    double currentOutputError = outputError.get(outImageIndex);
                                    gradients[kernelIndex].add(kernelIndices, currentOutputError * inputs.get(inImageIndex));
                                    gradients[kernelIndex + 1].add(biasIndex, currentOutputError);
                                }
                            }
                        }
                    }
                }
            }
        }
        return gradients;
    }

    /*
    @Override
    public Tensor getDelta(Tensor outputError) {
        delta = Tensor.zeros(1, IN_X, IN_Y, IN_IMAGES);
        for (int inF = 0; inF < IN_IMAGES; inF++) {
            for (int outF = 0; outF < NUM_FILTERS; outF++) {
                int kernelIndex = outF * 2;
                for (int x = 0; x < OUT_X; x++) {
                    for (int y = 0; y < OUT_Y; y++) {
                        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                                int[] inImageIndex = {0, x + kx, y + ky, inF};
                                int[] outImageIndex = {0, x, y, outF};
                                int[] kernelIndices = {kx, ky, inF};
                                if (!outOfBoundsForIn(inImageIndex[1], inImageIndex[2])) {
                                    double kernelValue = kernel[kernelIndex].get(kernelIndices);
                                    double currentDelta = outputError.get(outImageIndex) * kernelValue;
                                    delta.set(inImageIndex, delta.get(inImageIndex) + currentDelta);
                                }
                            }
                        }
                    }
                }
            }
        }
        return delta;
    }
    */
    @Override
    public Tensor getDelta(Tensor outputError) {
        delta = Tensor.zeros(1, IN_X, IN_Y, IN_IMAGES);
        for (int inF = 0; inF < IN_IMAGES; inF++) {
            for (int outF = 0; outF < NUM_FILTERS; outF++) {
                int kernelIndex = outF * 2;
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                        int[] kernelIndices = {kx, ky, inF};
                        double kernelValue = kernel[kernelIndex].get(kernelIndices);
                        for (int x = 0; x < OUT_X; x++) {
                            for (int y = 0; y < OUT_Y; y++) {
                                int[] inImageIndex = {0, x + kx, y + ky, inF};
                                int[] outImageIndex = {0, x, y, outF};
                                if (!outOfBoundsForIn(inImageIndex[1], inImageIndex[2])) {
                                    double currentDelta = outputError.get(outImageIndex) * kernelValue;
                                    delta.add(inImageIndex, currentDelta);
                                }
                            }
                        }
                    }
                }
            }
        }
        return delta;
    }


    private boolean outOfBoundsForIn(int x, int y) {
        return x < 0 || y < 0 || x >= IN_X || y >= IN_Y;
    }

    private boolean outOfBoundsForOut(int x, int y) {
        return x < 0 || y < 0 || x >= OUT_X || y >= OUT_Y;
    }
}
