package xenonn.sample;

public class Tensors {

    public static void main(String[] args) {
        long time;

        time = System.currentTimeMillis();
        //3873
        //
        for (int i = 0; i < 30000000; i++) {
            double d = Math.random() * 2 - 1;
            String s = Double.toString(d);
            String print = d < 0 ? s.substring(0, 3) : s.substring(0, 2);
        }
        System.out.println(System.currentTimeMillis() - time);

        time = System.currentTimeMillis();
        for (int i = 0; i < 30000000; i++) {
            double d = Math.random() * 2 - 1;
            String print = d < 0 ? Double.toString(d).substring(0, 3) : Double.toString(d).substring(0, 2);
        }
        System.out.println(System.currentTimeMillis() - time);

        time = System.currentTimeMillis();
        for (int i = 0; i < 30000000; i++) {
            double d = Math.random() * 2 - 1;
            int ind = d < 0 ? 2 : 3;
            String print = Double.toString(d).substring(0, ind);
        }
        System.out.println(System.currentTimeMillis() - time);

        time = System.currentTimeMillis();
        for (int i = 0; i < 30000000; i++) {
            double d = Math.random() * 2 - 1;
            String print = Double.toString(d).substring(0, d < 0 ? 2 : 3);
        }
        System.out.println(System.currentTimeMillis() - time);

    }
}
