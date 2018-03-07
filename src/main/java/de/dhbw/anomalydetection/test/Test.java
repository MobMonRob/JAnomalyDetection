package de.dhbw.anomalydetection.test;

/**
 *
 * @author rettig
 */
public class Test {
    public static void main(String[] args){
        double[][][] window = new double[1][2][3];
        System.out.println (window.length);
        System.out.println(window[0].length);
        System.out.println(window[0][0].length);
        // 1 2 3
        
        double[][] window1 = new double[3][4];
        System.out.println (window1.length);
        System.out.println(window1[0].length);
        // 3 4
    }
}
