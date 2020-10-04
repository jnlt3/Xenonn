package xenonn.nn.learningrule;

import xenonn.math.Tensor;
import xenonn.nn.layer.Layer;
import xenonn.nn.loss.Loss;
import xenonn.nn.model.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public abstract class GradientBasedOptimizer {

    protected final Model MODEL;
    protected final Loss LOSS;
    protected final Layer[] LAYERS;
    protected final int NUM_LAYERS;
    protected int capacity = -1;

    protected List<Tensor> trainIn;
    protected List<Tensor> trainOut;

    protected List<Tensor> testIn;
    protected List<Tensor> testOut;

    double loss;
    double numberOfValidOutputs;
    int correctPredictions;
    int numberOfPredictions;

    public GradientBasedOptimizer(Model model, Loss loss) {
        MODEL = model;
        LOSS = loss;
        LAYERS = MODEL.getLayers();
        NUM_LAYERS = LAYERS.length;

        trainIn = new ArrayList<>();
        trainOut = new ArrayList<>();
        testIn = new ArrayList<>();
        testOut = new ArrayList<>();
    }

    public void setCapacity(int capacity) {
        this.capacity = capacity;
    }

    public void setTrainSet(List<Tensor> in, List<Tensor> out) {
        if (in.size() != out.size()) {
            throw new RuntimeException("In: " + in.size() + "Out: " + out.size() + " don't match!");
        }
        this.trainIn = in;
        this.trainOut = out;
    }

    public void setTestSet(List<Tensor> in, List<Tensor> out) {
        if (in.size() != out.size()) {
            throw new RuntimeException("In: " + in.size() + "Out: " + out.size() + " don't match!");
        }
        this.testIn = in;
        this.testOut = out;
    }

    public void addTrainSet(List<Tensor> in, List<Tensor> out) {
        if (in.size() != out.size()) {
            throw new RuntimeException("In: " + in.size() + "Out: " + out.size() + " don't match!");
        }
        this.trainIn.addAll(in);
        this.trainOut.addAll(out);
        if (capacity != -1) {
            while (trainIn.size() > capacity) {
                trainIn.remove(0);
                trainOut.remove(0);
            }
        }
    }

    public void addTestSet(List<Tensor> in, List<Tensor> out) {
        if (in.size() != out.size()) {
            throw new RuntimeException("In: " + in.size() + "Out: " + out.size() + " don't match!");
        }
        this.testIn.addAll(in);
        this.testOut.addAll(out);
        if (capacity != -1) {
            while (testIn.size() > capacity) {
                testIn.remove(0);
                testOut.remove(0);
            }
        }
    }


    public void addTrain(Tensor in, Tensor out) {
        trainIn.add(in);
        trainOut.add(out);
        if (capacity != -1 && trainIn.size() > capacity) {
            trainIn.remove(0);
            trainOut.remove(0);
        }
    }

    public void addTest(Tensor in, Tensor out) {
        this.testIn.add(in);
        this.testOut.add(out);
    }

    public double trainLoss(int threads) {
        return loss(trainIn, trainOut, threads);
    }

    public double testLoss(int threads) {
        return loss(testIn, testOut, threads);
    }

    public double trainAccuracy(int threads) {
        return accuracy(trainIn, trainOut, threads);
    }

    public double testAccuracy(int threads) {
        return accuracy(testIn, testOut, threads);
    }

    private double loss(List<Tensor> in, List<Tensor> out, int threads) {
        loss = 0;
        numberOfValidOutputs = 0;
        if (threads > 1) {
            ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads - 1);
            int setPerThread = in.size() / threads;
            for (int i = 0; i < threads - 1; i++) {
                LossCalculator lossCalculator = new LossCalculator(in, out, i * setPerThread, (i + 1) * setPerThread);
                executor.submit(lossCalculator);
            }
            LossCalculator lossCalculator = new LossCalculator(in, out, setPerThread * (threads - 1), in.size());
            lossCalculator.run();
            while (executor.getActiveCount() != 0) ;
            return loss / numberOfValidOutputs;
        }
        LossCalculator lossCalculator = new LossCalculator(in, out, 0, in.size());
        lossCalculator.run();
        return loss / numberOfValidOutputs;
    }

    private double accuracy(List<Tensor> in, List<Tensor> out, int threads) {
        correctPredictions = 0;
        numberOfPredictions = 0;
        if (threads > 1) {
            ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads - 1);
            int setPerThread = in.size() / threads;
            for (int i = 0; i < threads - 1; i++) {
                AccuracyCalculator accuracyCalculator = new AccuracyCalculator(in, out, i * setPerThread, (i + 1) * setPerThread);
                executor.submit(accuracyCalculator);
            }
            AccuracyCalculator accuracyCalculator = new AccuracyCalculator(in, out, setPerThread * (threads - 1), in.size());
            accuracyCalculator.run();
            while (executor.getActiveCount() != 0) ;
            return correctPredictions * 1D / numberOfPredictions;
        }
        AccuracyCalculator accuracyCalculator = new AccuracyCalculator(in, out, 0, in.size());
        accuracyCalculator.run();
        return correctPredictions * 1D / numberOfPredictions;
    }

    public abstract void trainIterations(int iterations, int batchSize);

    public abstract void trainEpochs(int epochs, int batchSize);

    public abstract Tensor trainOnError(Tensor error, boolean returnInputError);

    public abstract void step();

    private static final int argmax(double[] d) {
        int index = 0;
        for (int i = 1; i < d.length; i++) {
            if (d[i] > d[index]) {
                index = i;
            }
        }
        return index;
    }

    private class LossCalculator implements Runnable {

        private final List<Tensor> IN;
        private final List<Tensor> OUT;
        private final int START_INDEX;
        private final int END_INDEX;

        public LossCalculator(List<Tensor> in, List<Tensor> out, int startIndex, int endIndex) {
            IN = in;
            OUT = out;
            START_INDEX = startIndex;
            END_INDEX = endIndex;
        }

        @Override
        public void run() {
            for (int i = START_INDEX; i < END_INDEX; i++) {
                Tensor modelOut = MODEL.feedForward(IN.get(i));
                Tensor expectedOut = OUT.get(i);
                if (expectedOut != null) {
                    loss += LOSS.getLoss(modelOut, expectedOut);
                    numberOfValidOutputs++;
                }
            }
        }
    }

    private class AccuracyCalculator implements Runnable {

        private final int[] ZERO = new int[2];
        private final List<Tensor> IN;
        private final List<Integer> OUT;
        private final int START_INDEX;
        private final int END_INDEX;

        public AccuracyCalculator(List<Tensor> in, List<Tensor> out, int startIndex, int endIndex) {
            IN = in;
            OUT = new ArrayList<>(out.size());
            for (int i = 0; i < out.size(); i++) {
                Tensor oneHot = out.get(i);
                if (oneHot != null) {
                    OUT.add(argmax(out.get(i).get(1, new int[2])));
                } else {
                    OUT.add(null);
                }
            }
            START_INDEX = startIndex;
            END_INDEX = endIndex;
        }

        @Override
        public void run() {
            for (int i = START_INDEX; i < END_INDEX; i++) {
                Integer expectedOut = OUT.get(i);
                if (expectedOut != null) {
                    if (argmax(MODEL.feedForward(IN.get(i)).get(1, ZERO)) == OUT.get(i)) {
                        correctPredictions++;
                    }
                    numberOfPredictions++;
                }
            }
        }
    }
}
