package xenonn.math;

import java.util.Arrays;

public class Tensor {

    private double[] value;
    private final int VALUE_LENGTH;
    private int[] shape;

    private Tensor(int... shape) {
        this.shape = shape;
        int valueLength = 1;
        for (int i = 0; i < this.shape.length; i++) {
            valueLength *= this.shape[i];
        }
        VALUE_LENGTH = valueLength;
        value = new double[VALUE_LENGTH];
    }

    @Override
    public String toString() {
        return "Tensor{ " +
                "\n VALUE=" + Arrays.toString(value) +
                ",\n SHAPE=" + Arrays.toString(shape) +
                "\n}";
    }

    public Tensor(double[] value, int... shape) {
        this.value = value;
        VALUE_LENGTH = this.value.length;
        this.shape = shape;
    }

    private int n21Dim(int... indices) {
        int n = 1;
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            index += indices[i] * n;
            n *= shape[i];
        }
        return index;
    }

    private int[] one2nDim(int index) {
        int n = VALUE_LENGTH;
        int[] indices = new int[shape.length];
        for (int i = indices.length - 1; i >= 0; i--) {
            n /= shape[i];
            indices[i] = index / n;
            index -= indices[i] * n;
        }
        return indices;
    }

    public Tensor copy() {
        return new Tensor(value.clone(), shape.clone());
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < tensor.value.length; i++) {
            tensor.value[i] = 1;
        }
        return tensor;
    }

    public static Tensor eye(int... shape) {
        if (shape.length != 2) {
            throw new RuntimeException("2 dimensions specifically are needed for eye");
        }
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                if (i == j) {
                    tensor.value[i + j * shape[0]] = 1;
                }
            }
        }
        return tensor;
    }

    public void reshape(int... shape) {
        int shapeSize = 1;
        for (int i = 0; i < shape.length; i++) {
            shapeSize *= shape[i];
        }
        if (shapeSize != VALUE_LENGTH) {
            throw new RuntimeException("This shape doesn't fit the tensor");
        }
        this.shape = new int[shape.length];
        System.arraycopy(shape, 0, this.shape, 0, shape.length);
    }

    public int[] getShape() {
        return shape;
    }

    public void set(int[] indices, double value) {
        this.value[n21Dim(indices)] = value;
    }

    public double get(int... indices) {
        return value[n21Dim(indices)];
    }

    public double[] get(int axis, int[] indices) {
        double[] values = new double[shape[axis]];
        int[] indicesCopy = indices.clone();
        for (int i = 0; i < shape[axis]; i++) {
            indicesCopy[axis] = i;
            values[i] = value[n21Dim(indicesCopy)];
        }
        return values;
    }

    public double[] getAll() {
        return value;
    }

    public int size() {
        return VALUE_LENGTH;
    }

    public void zero() {
        value = new double[VALUE_LENGTH];
    }

    public double sum() {
        double sum = 0;
        for (int i = 0; i < value.length; i++) {
            sum += value[i];
        }
        return sum;
    }

    public static double sum(Tensor tensor) {
        double sum = 0;
        for (int i = 0; i < tensor.value.length; i++) {
            sum += tensor.value[i];
        }
        return sum;
    }

    public Tensor add(double d) {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] += d;
        }
        return this;
    }

    public void add(int[] indices, double d) {
        value[n21Dim(indices)] += d;
    }

    public static Tensor add(Tensor tensor, double d) {
        Tensor returnTensor = zeros(tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] = tensor.value[i] + d;
        }
        return returnTensor;
    }

    public Tensor multiply(double d) {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] *= d;
        }
        return this;
    }

    public void multiply(int[] indices, double d) {
        value[n21Dim(indices)] *= d;
    }

    public static Tensor multiply(Tensor tensor, double d) {

        Tensor returnTensor = zeros(tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] = tensor.value[i] * d;
        }
        return returnTensor;
    }

    public Tensor divideByTensor(double d) {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] = d / value[i];
        }
        return this;
    }

    public static Tensor divideByTensor(double d, Tensor tensor) {
        Tensor returnTensor = zeros(tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] = d / tensor.value[i];
        }
        return returnTensor;
    }

    public Tensor square() {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] *= value[i];
        }
        return this;
    }

    public static Tensor square(Tensor tensor) {
        Tensor returnTensor = zeros(tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] = sq(tensor.value[i]);
        }
        return returnTensor;
    }

    public Tensor cube() {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] *= sq(value[i]);
        }
        return this;
    }

    public static Tensor cube(Tensor tensor) {
        Tensor returnTensor = new Tensor(tensor.value, tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] *= sq(tensor.value[i]);
        }
        return returnTensor;
    }

    public Tensor sqrt() {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] = Math.sqrt(value[i]);
        }
        return this;
    }

    public static Tensor sqrt(Tensor tensor) {
        Tensor returnTensor = new Tensor(tensor.value, tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] = Math.sqrt(tensor.value[i]);
        }
        return returnTensor;
    }

    public Tensor cbrt() {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] = Math.cbrt(value[i]);
        }
        return this;
    }

    public static Tensor cbrt(Tensor tensor) {
        Tensor returnTensor = new Tensor(tensor.value, tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] = Math.cbrt(tensor.value[i]);
        }
        return returnTensor;
    }

    public Tensor power(double power) {
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] = Math.pow(value[i], power);
        }
        return this;
    }

    public static Tensor power(Tensor tensor, double power) {
        Tensor returnTensor = new Tensor(tensor.value, tensor.shape);
        for (int i = 0; i < tensor.VALUE_LENGTH; i++) {
            returnTensor.value[i] = Math.pow(tensor.value[i], power);
        }
        return returnTensor;
    }

    public Tensor add(Tensor t) {
        if (t.shape.length != shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t.shape.length; i++) {
            if (t.shape[i] != shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] += t.value[i];
        }
        return this;
    }

    public static Tensor add(Tensor t0, Tensor t1) {
        if (t0.shape.length != t1.shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t0.shape.length; i++) {
            if (t0.shape[i] != t1.shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        Tensor returnTensor = new Tensor(t0.value.clone(), t0.shape);
        for (int i = 0; i < t0.VALUE_LENGTH; i++) {
            returnTensor.value[i] = t0.value[i] + t1.value[i];
        }
        return returnTensor;
    }

    public Tensor subtract(Tensor t) {
        if (t.shape.length != shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t.shape.length; i++) {
            if (t.shape[i] != shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] -= t.value[i];
        }
        return this;
    }

    public static Tensor subtract(Tensor t0, Tensor t1) {
        if (t0.shape.length != t1.shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t0.shape.length; i++) {
            if (t0.shape[i] != t1.shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        Tensor returnTensor = new Tensor(t0.value.clone(), t0.shape);
        for (int i = 0; i < t0.VALUE_LENGTH; i++) {
            returnTensor.value[i] = t0.value[i] - t1.value[i];
        }
        return returnTensor;
    }

    public static Tensor subtract(double d, Tensor t0) {
        Tensor returnTensor = new Tensor(t0.value.clone(), t0.shape);
        for (int i = 0; i < t0.VALUE_LENGTH; i++) {
            returnTensor.value[i] = d - t0.value[i];
        }
        return returnTensor;
    }

    public Tensor multiply(Tensor t) {
        if (t.shape.length != shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t.shape.length; i++) {
            if (t.shape[i] != shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] *= t.value[i];
        }
        return this;
    }

    public static Tensor multiply(Tensor t0, Tensor t1) {
        if (t0.shape.length != t1.shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t0.shape.length; i++) {
            if (t0.shape[i] != t1.shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        Tensor returnTensor = new Tensor(t0.value.clone(), t0.shape);
        for (int i = 0; i < t0.VALUE_LENGTH; i++) {
            returnTensor.value[i] = t0.value[i] * t1.value[i];
        }
        return returnTensor;
    }


    public Tensor divide(Tensor t) {
        if (t.shape.length != shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t.shape.length; i++) {
            if (t.shape[i] != shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        for (int i = 0; i < VALUE_LENGTH; i++) {
            value[i] /= t.value[i];
        }
        return this;
    }

    public static Tensor divide(Tensor t0, Tensor t1) {
        if (t0.shape.length != t1.shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        for (int i = 0; i < t0.shape.length; i++) {
            if (t0.shape[i] != t1.shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        Tensor returnTensor = new Tensor(t0.value.clone(), t0.shape);
        for (int i = 0; i < t0.VALUE_LENGTH; i++) {
            returnTensor.value[i] = t0.value[i] / t1.value[i];
        }
        return returnTensor;
    }

    public static Tensor log(Tensor tensor) {
        Tensor returnTensor = new Tensor(tensor.getShape());
        for (int i = 0; i < returnTensor.value.length; i++) {
            returnTensor.value[i] = Math.log(tensor.value[i]);
        }
        return returnTensor;
    }

    public static Tensor transpose(Tensor tensor) {
        int[] shape = shift(tensor.shape);
        Tensor transposedTensor = Tensor.zeros(shape);
        for (int i = 0; i < tensor.value.length; i++) {
            int index = transposedTensor.n21Dim(shift(tensor.one2nDim(i)));
            transposedTensor.value[index] = tensor.value[i];
        }
        return transposedTensor;
    }

    private static int[] shift(int[] indices) {
        int[] shiftedIndices = new int[indices.length];
        for (int i = 0; i < shiftedIndices.length - 1; i++) {
            shiftedIndices[i] = indices[i + 1];
        }
        shiftedIndices[shiftedIndices.length - 1] = indices[0];
        return shiftedIndices;
    }

    public static Tensor dotProduct(Tensor t0, Tensor t1) {
        int[] t0Shape = t0.shape;
        int[] t1Shape = t1.shape;
        int t0ShapeLength = t0Shape.length;
        int t1ShapeLength = t1Shape.length;
        int minLength = Math.min(t0ShapeLength, t1ShapeLength);

        int t0Reshape = t0Shape[0];
        int t1Reshape = t1Shape[0];
        for (int i = 1; i < minLength - 1; i++) {
            t0Reshape *= t0Shape[i];
            t1Reshape *= t1Shape[i];
        }
        int[] t0TrimmedShape = new int[t0ShapeLength - minLength + 2];
        int[] t1TrimmedShape = new int[t1ShapeLength - minLength + 2];
        t0TrimmedShape[0] = t0Reshape;
        t1TrimmedShape[0] = t1Reshape;
        for (int i = 1; i < t0TrimmedShape.length; i++) {
            t0TrimmedShape[i] = t0Shape[i + minLength - 2];
        }
        for (int i = 1; i < t1TrimmedShape.length; i++) {
            t1TrimmedShape[i] = t1Shape[i + minLength - 2];
        }

        int[] t0NewShape = {t0TrimmedShape[0], 1};
        int[] t1NewShape = {t1TrimmedShape[0], 1};

        for (int i = 1; i < t0TrimmedShape.length; i++) {
            t0NewShape[1] *= t0TrimmedShape[i];
        }

        for (int i = 1; i < t1TrimmedShape.length; i++) {
            t1NewShape[1] *= t1TrimmedShape[i];
        }

        Tensor t2 = new Tensor(t0.value, t0NewShape);
        Tensor t3 = new Tensor(t1.value, t1NewShape);

        if (t0NewShape[0] == 1) {
            double[] value = new double[t1NewShape[1]];
            for (int i = 0; i < t1NewShape[0]; i++) {
                for (int j = 0; j < t1NewShape[1]; j++) {
                    value[j] += t2.get(0, i) * t3.get(i, j);
                }
            }
            int[] reshape = new int[t1Shape.length - 1];
            System.arraycopy(t1Shape, 1, reshape, 0, t1ShapeLength - 1);
            return new Tensor(value, 1, value.length);
        } else if (t0NewShape[1] == 1) {
            Tensor value = zeros(t0NewShape[0], t1NewShape[1]);
            for (int i = 0; i < t1NewShape[0]; i++) {
                for (int j = 0; j < t0NewShape[0]; j++) {
                    for (int k = 0; k < t1NewShape[1]; k++) {
                        value.set(new int[]{j, k}, value.get(j, k) + t2.get(j, 0) * t3.get(i, k));
                    }
                }
            }
            int[] reshape = new int[t1Shape.length];
            System.arraycopy(t1Shape, 1, reshape, 1, t0Shape.length - 1);
            reshape[0] = t0NewShape[0];
            value.reshape(reshape);
            return value;
        }
        return multiply(t2, t3);
    }

    //TODO:fix concatenation shape[shape(1, 2, 3), shape(2, 2, 3)] = shape(3, 2, 3)
    public Tensor concat(Tensor t) {
        if (shape.length != t.shape.length) {
            throw new RuntimeException("Tensors don't match");
        }
        int index = shape.length - 1;
        for (int i = 0; i < index; i++) {
            if (shape[i] != t.shape[i]) {
                throw new RuntimeException("Tensors don't match");
            }
        }
        shape[index] += t.shape[index];
        double[] newValue = new double[value.length + t.value.length];
        System.arraycopy(value, 0, newValue, 0, value.length);
        System.arraycopy(t.value, 0, newValue, value.length, t.value.length);
        value = newValue;
        return this;
    }

    public static Tensor concat(Tensor t0, Tensor t1) {
        Tensor returnTensor = t0.copy();
        returnTensor.concat(t1);
        return returnTensor;
    }

    public static Tensor[] split(Tensor t, int split0) {
        double[] tValue = t.value;
        int[] tShape = t.shape;
        int[] t0Shape = tShape.clone();
        int[] t1Shape = tShape.clone();
        int lastIndex = t0Shape.length - 1;
        int sharedShapeSize = tValue.length / tShape[lastIndex];
        t0Shape[lastIndex] = split0;
        t1Shape[lastIndex] -= split0;
        double[] t0Value = new double[sharedShapeSize * t0Shape[lastIndex]];
        double[] t1Value = new double[sharedShapeSize * t1Shape[lastIndex]];
        System.arraycopy(tValue, 0, t0Value, 0, t0Value.length);
        System.arraycopy(tValue, t0Value.length, t1Value, 0, t1Value.length);
        Tensor splitT0 = new Tensor(t0Value, t0Shape);
        Tensor splitT1 = new Tensor(t1Value, t1Shape);
        return new Tensor[]{splitT0, splitT1};
    }

    public double mean() {
        return sum() / VALUE_LENGTH;
    }


    public double variance() {
        double mean = mean();
        double variance = 0;
        for (int i = 0; i < VALUE_LENGTH; i++) {
            variance += sq(mean - value[i]);
        }
        return variance / (VALUE_LENGTH - 1);
    }

    public static double mean(Tensor tensor) {
        return tensor.mean();
    }

    public static double variance(Tensor tensor) {
        return tensor.variance();
    }


    private static double sq(double x) {
        return x * x;
    }
}
