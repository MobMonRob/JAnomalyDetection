package de.dhbw.anomalydetection.util;

import org.apache.commons.math3.stat.StatUtils;

/**
 *
 * @author Oliver Rettig
 */
public class MathUtils {
    
    /**
    * Normalize by subtraction of the trials mean and division by std.
    * 
    * @param multidimwindow [BATCH_SIZE][dim][TRIAL_WINDOW_SIZE]
    * @return [BATCH_SIZE][dim][TRIAL_WINDOW_SIZE]
    */
   public static double[][][] normalize(double[][][] multidimwindow){
       // iterate over dim
       for (int d=0;d<multidimwindow[0].length;d++){
           // iterate over batch size
           for (int batch=0;batch<multidimwindow.length;batch++){
                multidimwindow[batch][d] = StatUtils.normalize(multidimwindow[batch][d]);
           }
       }
       return multidimwindow;
   }
    /**
    * Normalize by subtraction of the trials mean and division by std.
    * 
    * @param multidimwindow [dim][TRIAL_WINDOW_SIZE]
    * @return [dim][TRIAL_WINDOW_SIZE]
    */
   public static double[][] normalize(double[][] multidimwindow){
       // iterate over dim
       for (int d=0;d<multidimwindow.length;d++){
                multidimwindow[d] = StatUtils.normalize(multidimwindow[d]);
       }
       return multidimwindow;
   }
   
    /**
    * fft.
    * 
    * After our tumbling count window is filled we apply fast Fourier 
    * transformation (FFT) to obtain the frequency spectrum of the signals.<p>
    * 
    * @param x input
    * @return fft
    */
   /*public static double[] fft(double[] x){
        DoubleFFT_1D fftDo = new DoubleFFT_1D(x.length);
        double[] fft = new double[x.length];
        System.arraycopy(x, 0, fft, 0, x.length);
        fftDo.realForward(fft);
        return fft;
   }*/
}
