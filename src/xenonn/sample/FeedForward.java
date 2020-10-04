package xenonn.sample;

import xenonn.math.Tensor;
import xenonn.nn.activation.ReLU;
import xenonn.nn.layer.Activation;
import xenonn.nn.layer.Dense;
import xenonn.nn.layer.Softmax;
import xenonn.nn.learningrule.BackPropagation;
import xenonn.nn.learningrule.GradientBasedOptimizer;
import xenonn.nn.loss.MSELoss;
import xenonn.nn.model.Model;
import xenonn.nn.model.Sequential;
import xenonn.nn.optimizer.Optimizer;
import xenonn.nn.optimizer.RMSProp;

import java.util.*;

public class FeedForward {

    private static List<Tensor> in;
    private static List<Tensor> out;
    private static final double HALF_PI = Math.PI / 2;

    public static void main(String[] args) {
        in = new ArrayList<>();
        out = new ArrayList<>();

        in.add(new Tensor(new double[]{0, 0}, 1, 2));
        in.add(new Tensor(new double[]{0, 1}, 1, 2));
        in.add(new Tensor(new double[]{1, 0}, 1, 2));
        in.add(new Tensor(new double[]{1, 1}, 1, 2));
        out.add(new Tensor(new double[]{1, 0}, 1, 2));
        out.add(new Tensor(new double[]{0, 1}, 1, 2));
        out.add(new Tensor(new double[]{0, 1}, 1, 2));
        out.add(new Tensor(new double[]{1, 0}, 1, 2));


        Model model = new Sequential(new int[]{1, 2},
                new Dense(64),
                new Activation(new ReLU()),
                new Dense(64),
                new Activation(new ReLU()),
                new Dense(2),
                new Softmax()
        );

        model.initialize();
        Optimizer optimizer = new RMSProp(model, 1e-3, 0.999);
        GradientBasedOptimizer backProp = new BackPropagation(model, optimizer, new MSELoss());

        backProp.setTrainSet(in, out);

        long time = System.nanoTime();
        backProp.trainIterations(100000, 1);
        long elapsed = System.nanoTime() - time;
        System.out.println(elapsed);

        System.out.println("result: ");
        for (int i = 0; i < in.size(); i++) {
            System.out.println(model.feedForward(in.get(i)));
        }

    }
}
