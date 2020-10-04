package xenonn.sample;

import xenonn.math.Tensor;
import xenonn.nn.activation.Sigmoid;
import xenonn.nn.layer.Activation;
import xenonn.nn.layer.Dense;
import xenonn.nn.learningrule.BackPropagation;
import xenonn.nn.learningrule.GradientBasedOptimizer;
import xenonn.nn.loss.MSELoss;
import xenonn.nn.model.AutoEncoder;
import xenonn.nn.optimizer.Adam;
import xenonn.nn.optimizer.Optimizer;

import java.util.ArrayList;
import java.util.List;

public class Encode {

    private static List<Tensor> in;
    private static List<Tensor> out;

    public static void main(String[] args) {
        in = new ArrayList<>();
        out = new ArrayList<>();

        in.add(new Tensor(new double[]{0, 0, 0}, 1, 3));
        in.add(new Tensor(new double[]{0, 1, 1}, 1, 3));
        in.add(new Tensor(new double[]{1, 0, 0}, 1, 3));
        in.add(new Tensor(new double[]{1, 1, 1}, 1, 3));
        AutoEncoder autoEncoder = new AutoEncoder(
                new int[]{1, 3},
                new Dense(2),
                new Activation(new Sigmoid()),
                new Dense(2),
                new Activation(new Sigmoid()),
                new Dense(3)
        );
        autoEncoder.initialize();
        System.out.println(autoEncoder.summary());
        Optimizer optimizer = new Adam(autoEncoder, 1e-3, 0.9, 0.999);
        GradientBasedOptimizer backPropagation = new BackPropagation(autoEncoder, optimizer, new MSELoss());
        backPropagation.setTrainSet(in, in);
        backPropagation.trainEpochs(10000, 1);


        System.out.println("result: ");
        for (int i = 0; i < in.size(); i++) {
            System.out.println("example: " + i);
            System.out.println(in.get(i));
            System.out.println(autoEncoder.encode(in.get(i), 2));
            System.out.println(autoEncoder.feedForward(in.get(i)));
        }
    }
}
