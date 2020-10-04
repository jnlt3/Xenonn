package xenon.sample;

import xenon.math.Tensor;
import xenon.nn.layer.Dense;
import xenon.nn.layer.recurrent.GRU;
import xenon.nn.layer.recurrent.LSTM;
import xenon.nn.learningrule.BackPropagationMemory;
import xenon.nn.learningrule.GradientBasedOptimizer;
import xenon.nn.loss.MSELoss;
import xenon.nn.model.Model;
import xenon.nn.model.Sequential;
import xenon.nn.optimizer.*;

import java.util.ArrayList;
import java.util.List;

public class RecurrentNet {

    private static List<Tensor> trainIn;
    private static List<Tensor> trainOut;
    private static final int LENGTH = 1;

    public static void main(String[] args) {
        trainIn = new ArrayList<>();
        trainOut = new ArrayList<>();

        double[] sums = new double[LENGTH];
        for (int i = 0; i < 5; i++) {
            double[] in = new double[LENGTH];
            for (int j = 0; j < LENGTH; j++) {
                in[j] = 1 / 5D;
                sums[j] += in[j];
            }
            trainIn.add(new Tensor(in, 1, LENGTH));
            if (i == 4) {
                trainOut.add(new Tensor(sums.clone(), 1, LENGTH));
            } else {
                trainOut.add(null);
            }
        }


        GRU recurrent0 = new GRU(3);
        Model model = new Sequential(new int[]{1, LENGTH},
                new Dense(LENGTH),
                recurrent0,
                new Dense(LENGTH));

        model.initialize();
        System.out.println(model.summary());

        Optimizer optimizer = new Adam(model, 1e-3, 0.9, 0.99);
        GradientBasedOptimizer backPropagation = new BackPropagationMemory(model, optimizer, new MSELoss());

        backPropagation.setTrainSet(trainIn, trainOut);


        for (int i = 0; i < trainIn.size(); i++) {
            System.out.println("out: ");
            System.out.println(model.feedForward(trainIn.get(i)));
            System.out.println("expected: ");
            System.out.println(trainOut.get(i));
        }
        recurrent0.resetMemory();
        System.out.println(backPropagation.trainLoss(1));
        recurrent0.resetMemory();

        long time = System.currentTimeMillis();
        //19708
        for (int i = 0; i < 300; i++) {
            backPropagation.trainIterations(100, 1);
            /*
            recurrent0.resetMemory();
            System.out.println(backPropagation.trainLoss(1));
            recurrent0.resetMemory();

             */
        }
        System.out.println(System.currentTimeMillis() - time);
        System.out.println("result: ");
        for (int i = 0; i < trainIn.size(); i++) {
            System.out.println("out: ");
            System.out.println(model.feedForward(trainIn.get(i)));
            System.out.println("expected: ");
            System.out.println(trainOut.get(i));
        }

    }
}
