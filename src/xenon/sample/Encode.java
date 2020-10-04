package xenon.sample;

import xenon.math.Tensor;
import xenon.nn.activation.Sigmoid;
import xenon.nn.layer.Activation;
import xenon.nn.layer.Dense;
import xenon.nn.learningrule.BackPropagation;
import xenon.nn.learningrule.GradientBasedOptimizer;
import xenon.nn.loss.MSELoss;
import xenon.nn.model.AutoEncoder;
import xenon.nn.model.Model;
import xenon.nn.optimizer.Adam;
import xenon.nn.optimizer.Optimizer;
import xenon.nn.optimizer.RMSProp;

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
