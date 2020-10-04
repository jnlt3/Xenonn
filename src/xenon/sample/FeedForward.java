package xenon.sample;

import xenon.math.Tensor;
import xenon.nn.activation.ReLU;
import xenon.nn.activation.Sigmoid;
import xenon.nn.layer.Activation;
import xenon.nn.layer.Dense;
import xenon.nn.layer.Softmax;
import xenon.nn.learningrule.BackPropagation;
import xenon.nn.learningrule.GradientBasedOptimizer;
import xenon.nn.loss.CrossEntropyLoss;
import xenon.nn.loss.MSELoss;
import xenon.nn.model.Model;
import xenon.nn.model.Sequential;
import xenon.nn.optimizer.Adam;
import xenon.nn.optimizer.Optimizer;
import xenon.nn.optimizer.RMSProp;

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
