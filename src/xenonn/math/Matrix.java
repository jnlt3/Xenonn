package xenonn.math;

import java.util.Arrays;

public class Matrix {

    private double[][] value;
    private int rows;
    private int columns;

    public Matrix(double[][] value) {
        this.value = value.clone();
        rows = value.length;
        columns = value[0].length;
    }

    public Matrix(double[] value, int axis) {
        if (axis == 0) {
            rows = 1;
            columns = value.length;
            this.value = new double[rows][columns];
            System.arraycopy(value, 0, this.value[0], 0, columns);
        } else if (axis == 1) {
            rows = value.length;
            columns = 1;
            this.value = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                this.value[i][0] = value[i];
            }
        } else {
            throw new RuntimeException("Axis can either be 0 or 1");
        }
    }

    @Override
    public String toString() {
        return "Matrix{" +
                "value=" + Arrays.toString(value) +
                '}';
    }

    public Matrix(int rows, int columns) {
        value = new double[rows][columns];
        this.rows = rows;
        this.columns = columns;
    }

    public double[] getRow(int row) {
        return value[row];
    }

    public double[] getColumn(int col) {
        double[] column = new double[rows];
        for (int i = 0; i < col; i++) {
            column[i] = value[i][col];
        }
        return column;
    }

    public Matrix copy() {
        double[][] matrix = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(value[i], 0, matrix[i], 0, columns);
        }
        return new Matrix(matrix);
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public double[][] getValue() {
        return value;
    }

    public int size() {
        return rows * columns;
    }

    public void assign(double[][] value) {
        this.value = value.clone();
    }

    public double get(int row, int column) {
        return value[row][column];
    }

    public void set(int row, int column, double value) {
        this.value[row][column] = value;
    }

    public double sum() {
        double sum = 0;
        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                sum += value[i][j];
            }
        }
        return sum;
    }

    public double[] sum(int dimension) {
        double[] sum;
        switch (dimension) {
            case 0:
                sum = new double[rows];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < columns; j++) {
                        sum[i] += value[i][j];
                    }
                }
                break;
            case 1:
                sum = new double[columns];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < columns; j++) {
                        sum[j] += value[i][j];
                    }
                }
                break;
            default:
                throw new RuntimeException("dimension must be 0 or 1");
        }
        return sum;
    }

    public void zero() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                value[i][j] = 0;
            }
        }
    }

    public static Matrix zeros(int rows, int columns) {
        double[][] value = new double[rows][columns];
        return new Matrix(value);
    }

    public static Matrix ones(int rows, int columns) {
        double[][] value = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                value[i][j] = 1;
            }
        }
        return new Matrix(value);
    }


    public static Matrix eye(int rows) {
        double[][] value = new double[rows][rows];
        for (int i = 0; i < rows; i++) {
            value[i][i] = 1;
        }
        return new Matrix(value);
    }

    public static Matrix square(Matrix m) {
        double[][] squaredValue = new double[m.rows()][m.columns()];
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.columns(); j++) {
                squaredValue[i][j] = m.value[i][j] * m.value[i][j];
            }
        }
        return new Matrix(squaredValue);
    }

    public static Matrix divide(Matrix m, double divisor) {
        double[][] dividedValue = new double[m.rows()][m.columns()];
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.columns(); j++) {
                dividedValue[i][j] = m.value[i][j] / divisor;
            }
        }
        return new Matrix(dividedValue);
    }

    public static Matrix multiply(Matrix m, double multiplier) {
        double[][] multipliedValue = new double[m.rows()][m.columns()];
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.columns(); j++) {
                multipliedValue[i][j] = m.value[i][j] * multiplier;
            }
        }
        return new Matrix(multipliedValue);
    }


    public static Matrix getSubMatrix(Matrix m, int x, int y, int rows, int columns) {
        double[][] value = new double[rows][columns];
        for (int j = y; j < y + columns; j++) {
            for (int i = x; i < x + rows; i++) {
                value[i - x][j - x] = m.getValue()[i][j];
            }
        }
        return new Matrix(value);
    }

    public static Matrix transpose(Matrix m) {
        double[][] transposedValue = new double[m.columns()][m.rows()];
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.columns(); j++) {
                transposedValue[j][i] = m.value[i][j];
            }
        }
        return new Matrix(transposedValue);
    }

    public void add(Matrix m1) {
        if (m1.rows() == rows() && m1.columns() == columns()) {
            int rows = m1.rows();
            int columns = m1.columns();
            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    value[i][j] += m1.get(i, j);
                }
            }
        } else {
            throw new RuntimeException("number of rows and columns must be equal");
        }
    }

    public static Matrix add(Matrix m1, Matrix m2) {
        if (m1.rows() == m2.rows() && m1.columns() == m2.columns()) {
            int rows = m1.rows();
            int columns = m1.columns();
            double[][] value = new double[rows][columns];
            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    value[i][j] = m1.get(i, j) + m2.get(i, j);
                }
            }
            return new Matrix(value);
        } else {
            throw new RuntimeException("number of rows and columns must be equal");
        }
    }

    public static Matrix add(Matrix m1, double m) {
        int rows = m1.rows();
        int columns = m1.columns();
        double[][] value = new double[rows][columns];
        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                value[i][j] = m1.get(i, j) + m;
            }
        }
        return new Matrix(value);
    }


    public void subtract(Matrix matrix) {
        if (rows() == matrix.rows() && columns() == matrix.columns()) {
            int rows = rows();
            int columns = columns();
            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    value[i][j] -= matrix.get(i, j);
                }
            }
        } else {
            throw new RuntimeException("number of rows and columns must be equal");
        }
    }

    public static Matrix subtract(Matrix m1, Matrix m2) {
        if (m1.rows() == m2.rows() && m1.columns() == m2.columns()) {
            int rows = m1.rows();
            int columns = m1.columns();
            double[][] value = new double[rows][columns];
            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    value[i][j] = m1.get(i, j) - m2.get(i, j);
                }
            }
            return new Matrix(value);
        } else {
            throw new RuntimeException("number of rows and columns must be equal");
        }
    }

    public static Matrix dotProduct(Matrix m1, Matrix m2) {

        if (m1.rows() == 1) {
            double[] value = new double[m2.columns()];
            for (int i = 0; i < m2.rows(); i++) {
                for (int k = 0; k < m2.columns(); k++) {
                    value[k] += m1.get(0, i) * m2.get(i, k);
                }
            }
            return new Matrix(value, 0);
        } else if (m1.columns() == 1) {
            double[][] value = new double[m1.rows()][m2.columns()];
            for (int i = 0; i < m2.rows(); i++) {
                for (int j = 0; j < m1.rows(); j++) {
                    for (int k = 0; k < m2.columns(); k++) {
                        value[j][k] += m1.get(j, 0) * m2.get(i, k);
                    }
                }
            }
            return new Matrix(value);
        }
        return multiply(m1, m2);
    }

    public void multiply(double d) {
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < columns(); j++) {
                value[i][j] *= d;
            }
        }
    }

    public static Matrix multiply(Matrix m1, Matrix m2) {
        double[][] value = new double[m1.rows()][m1.columns()];
        for (int i = 0; i < m1.rows(); i++) {
            for (int j = 0; j < m1.columns(); j++) {
                value[i][j] = m1.get(i, j) * m2.get(i, j);
            }
        }
        return new Matrix(value);
    }

    public static Matrix divide(Matrix m1, Matrix m2) {
        double[][] value = new double[m1.rows()][m1.columns()];
        for (int i = 0; i < m1.rows(); i++) {
            for (int j = 0; j < m1.columns(); j++) {
                value[i][j] = m1.get(i, j) / m2.get(i, j);
            }
        }
        return new Matrix(value);
    }

    public static Matrix sqrt(Matrix m1) {
        double[][] value = new double[m1.rows()][m1.columns()];
        for (int i = 0; i < m1.rows(); i++) {
            for (int j = 0; j < m1.columns(); j++) {
                value[i][j] = Math.sqrt(m1.get(i, j));
            }
        }
        return new Matrix(value);
    }


    public static Matrix product(Matrix m1, Matrix m2) {
        double[][] value = new double[m1.rows()][m2.columns()];
        for (int i = 0; i < m1.rows(); i++) {
            for (int j = 0; j < m2.columns(); j++) {
                for (int k = 0; k < m2.rows(); k++) {
                    value[i][j] += m1.getValue()[i][k] * m2.getValue()[k][j];
                }
            }
        }
        return new Matrix(value);
    }

    public static Matrix invert(Matrix m) {
        int len = m.rows();
        double[][] output = new double[len][len];
        double[][] a = new double[len][len];
        int index[] = new int[len];
        for (int i = 0; i < len; ++i) {
            a[i][i] = 1;
        }

        gaussian(m, index);

        for (int i = 0; i < len - 1; ++i) {
            for (int j = i + 1; j < len; ++j) {
                for (int k = 0; k < len; ++k) {
                    a[index[j]][k] -= m.getValue()[index[j]][i] * a[index[i]][k];
                }
            }
        }

        for (int i = 0; i < len; ++i) {
            output[len - 1][i] = a[index[len - 1]][i] / m.getValue()[index[len - 1]][len - 1];
            for (int j = len - 2; j >= 0; --j) {
                output[j][i] = a[index[j]][i];
                for (int k = j + 1; k < len; ++k) {
                    output[j][i] -= m.getValue()[index[j]][k] * output[k][i];
                }
                output[j][i] /= m.getValue()[index[j]][j];
            }
        }
        return new Matrix(output);
    }

    public double mean() {
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sum += value[i][j];
            }
        }
        return sum / (rows * columns);
    }


    public double variance() {
        double mean = mean();
        double variance = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                variance += sq(mean - value[i][j]);
            }
        }
        return variance / (rows * columns - 1);
    }

    private static void gaussian(Matrix m, int index[]) {
        int n = index.length;
        double c[] = new double[n];
        for (int i = 0; i < n; i++) {
            index[i] = i;
        }

        for (int i = 0; i < n; i++) {
            double c1 = 0;
            for (int j = 0; j < n; j++) {
                double c0 = Math.abs(m.getValue()[i][j]);
                if (c0 > c1) {
                    c1 = c0;
                }
            }
            c[i] = c1;
        }

        int k = 0;
        for (int j = 0; j < n - 1; j++) {
            double pi1 = 0;

            for (int i = j; i < n; i++) {
                double pi0 = Math.abs(m.getValue()[index[i]][j]);
                pi0 /= c[index[i]];
                if (pi0 > pi1) {
                    pi1 = pi0;
                    k = i;
                }
            }

            int itmp = index[j];
            index[j] = index[k];
            index[k] = itmp;

            for (int i = j + 1; i < n; ++i) {
                double pj = m.getValue()[index[i]][j] / m.getValue()[index[j]][j];
                m.set(index[i], j, pj);

                for (int l = j + 1; l < n; ++l) {
                    m.set(index[i], l, m.getValue()[index[i]][l] - pj * m.getValue()[index[j]][l]);
                }
            }

        }
    }

    private static double sq(double x) {
        return x * x;
    }
}
