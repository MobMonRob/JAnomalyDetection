package de.dhbw.anomalydetection;

import de.dhbw.anomalydetection.util.PlotUtils;
import de.dhbw.anomalydetection.filter.csv.CSVDataReader;
import de.dhbw.anomalydetection.util.MathUtils;
import java.util.Random;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.FileNotFoundException;
import java.io.IOException;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
        
import org.deeplearning4j.ui.api.UIServer;

import java.io.File;

import java.util.logging.Level;
import java.util.logging.Logger;

import org.deeplearning4j.util.ModelSerializer;

/**
 * Note: The training and anomaly detection is taking place at the same time, 
 * because the neural network continuously learns what normal data looks like 
 * and after it sees anomalies, it will raise an error.
 * 
 * https://www.ibm.com/developerworks/analytics/library/iot-deep-learning-anomaly-detection-3/index.html<br>
 * https://github.com/romeokienzler/dl4j-examples/blob/master/dl4j-examples-scala/src/main/scala/IoTAnomalyExampleLSTMFFTWatsonIoT.scala<p>
 * 
 * TODO<br>
 * - Aktivierungsfunktion tanh da dies die Ausreißer betont?<p>
 * 
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/anomalydetection/VaeMNISTAnomaly.java
 * --> Hinweis zum VariationalAutoencoder für Anomalydetection aber vermutlich nicht für Zeitreihen
 * 
 * @author Oliver Rettig (based on the IoTAnomalyExampleLSTMFFTWatsonIoT of Romeo Kienzler) 
 */
public class AnomalyDetectorTest {
    
   // Random number generator seed, for reproducability
   private static final int seed = 12345;
    
   private static final int SLIDING_WINDOW_SIZE = 100;
     
   private static final int TEST_WINDOW_SIZE = 560; 
  
   private static int frameIndex = 0; // processed frame index inside a window
   
   
   private static int dim = 1; // dimensionality of the input data, defined by the JsonDataMovingWindow     
      
   
   private static MultiLayerNetwork net;
   private static AnomalyDetectorTest detector;
      
   private final Random rng = new Random(seed);
   
   private org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae;
   
   public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException {
       detector = new AnomalyDetectorTest();
       test(true);
       while(true);
   }
   
   /**
    * Constructor.
    * 
    * @param channels dimensionality of the data
    */
   private AnomalyDetectorTest(){
       
        net = loadModel();
        
        // falls VAE dann muss ich das layer noch rausholen...
        Object obj = net.getLayer(1);
        if (obj instanceof org.deeplearning4j.nn.layers.variational.VariationalAutoencoder){
            vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) obj;
            System.out.println("VAE model found...");
        } else {
            System.out.println("Common autoencoder model found...");
        }
        // add an listener which outputs the error every 1 parameter updates
        //net.setListeners(new ScoreIterationListener(1));   
      
   }
   
   /**
    * Test the net by timeseries with single step method.
    * 
    * @param stepByStep
    * @throws FileNotFoundException
    * @throws IOException
    * @throws InterruptedException 
    */
   public static void test(boolean stepByStep) throws FileNotFoundException, IOException, InterruptedException {
     
        CSVDataReader data = new CSVDataReader(TEST_WINDOW_SIZE); 
        double[][] window = new double[dim][TEST_WINDOW_SIZE];
        double[][] window2 = new double[dim][TEST_WINDOW_SIZE];
        
        // über die frames aller Dateien iterieren, eventuell wird bei einer Datei
        // bereits vor Ende abgebrochen
        while (data.hasNext()){
            String currentFileName = data.getFileName(); // beim ersten frame des ersten trials "unknown file"
            double[] frame = data.next();

            if (frameIndex < window[0].length && frameIndex < data.getCurrentFileRows()){
                    for (int i=0;i<dim;i++){
                        window[i][frameIndex] = frame[i];
                        window2[i][frameIndex] = frame[i];
                    }
                    frameIndex++;
            } else {
                if (window[0].length != data.getCurrentFileRows()){
                    System.out.println("The file \""+currentFileName+"\" has "+
                            data.getCurrentFileRows()+" rows, but windowsize is "+String.valueOf(TEST_WINDOW_SIZE));
                }
                double[][] result;
                window = MathUtils.normalize(window);
                window2 = MathUtils.normalize(window2);
                if (stepByStep){
                    result = detector.scoreTimeSerieWindowStepByStep(window);
                } else {
                    result = detector.scoreTimeSerieWindowComplete(window);
                }
                double[][] plot = new double[2][window[0].length];
                // alle frames eines windows durchiterieren
                double max0 = -Double.MAX_VALUE;
                double max1 = -Double.MAX_VALUE;
                for (int i=0;i<window[0].length;i++){
                    plot[0][i] = window2[0][i];
                    plot[1][i] = result[0][i];
                    if (plot[0][i] > max0) max0 = plot[0][i];
                    if (plot[1][i] > max1) max1 = plot[1][i];
                }
                for (int i=0;i<window[0].length;i++){
                    plot[1][i] *= max0/max1; 
                }
                PlotUtils.createChart(data.getFileName(), plot, new String[]{"in", "pred"});
                
                // reconstruction propability
                //plot = new double[1][window[0].length];
                //for (int i=0;i<window[0].length;i++){
                //     plot[0][i] = result[0][i];
                //}
                //PlotUtils.createChart(data.getFileName(), plot, new String[]{"prob"});
                
                // err = pred - in
                double[][] plot2 = new double[1][window[0].length];
                for (int i=0;i<window[0].length;i++){
                    plot2[0][i] = (result[0][i]-window2[0][i]);
                    plot2[0][i] *= plot2[0][i];
                }
                PlotUtils.createChart(data.getFileName(), plot2, new String[]{"MSerr"});
                
                //System.out.println(data.getFileName()+": window "+currentWindow+", score = "+score);
                frameIndex = 0;
                //currentTrial++;
            }
        }
   }
   /**
    * Process a timeserie window step by step to determine a score for each timestep.
    * 
    * siehe Beispiel für runTimeStep() und RNN networks:<br>
    * http://progur.com/2017/06/how-to-create-lstm-rnn-deeplearning4j.html<p>
    * 
    * @param timeSerieWindow [dims][frames]
    * @return score[dims][frames]
    */
   private double[][] scoreTimeSerieWindowStepByStep(double[][] timeSerieWindow){
        double[][] result = new double[timeSerieWindow.length][timeSerieWindow[0].length];
        net.rnnClearPreviousState();
        // frames durchiterieren
        for (int f=0;f<timeSerieWindow[0].length;f++){
            // multiple rows (dimension 0 in the input data) are used for multiple examples.
            // For a single time step prediction: the data is 2 dimensional, 
            // with shape [numExamples,nIn]; in this case, the output is also 2 
            // dimensional, with shape [numExamples,nOut]
            // For multiple time step predictions: the data is 3 dimensional, 
            // with shape [numExamples,nIn,numTimeSteps]; the output will have 
            // shape [numExamples,nOut,numTimeSteps]. 
            double[][] step = new double[1][timeSerieWindow.length]; // 1, dim=1
            // iteration over dim
            for (int i=0;i<timeSerieWindow.length;i++){
                step[0][i] = timeSerieWindow[i][f];
            }
            // returns output activations for the current frame == prediction
            // output for the rnnTimeStep and the output/feedForward methods 
            // should be identical (for each time step), whether we make these 
            // predictions all at once (output/feedForward) or whether these 
            // predictions are generated one or more steps at a time (rnnTimeStep). 
            // Thus, the only difference should be the computational cost.
            // https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/encdec/EncoderDecoderLSTM.java
            INDArray prediction = net.rnnTimeStep(Nd4j.create(step));
            
            if (vae == null){
                // ob das mit mehreren Dimensionen wirklich auch geht ist noch zu testen
                //FIXME
                for (int i=0;i<timeSerieWindow.length;i++){
                    // https://github.com/IsaacChanghau/StockPrediction/blob/master/src/main/java/com/isaac/stock/predict/StockPricePrediction.java
                    // getDouble(index des letzten Werts der Zeitreihe?) das legt obiges Beispiel nahe, passt aber nicht zu den Arraygrößen die ich hier sehe
                    result[i][f] = prediction.getDouble(i); 
                }
                
            // mit variational autoencoder
            } else {
                
                // https://github.com/skeydan/dl4j_in_action/blob/master/src/main/java/com/trivadis/deeplearning/VAEAnomalyDetectorMnist.java
                // --> beispiel für reconstructionLogPropability
                
                // vae.score(); ist immer == 0 
                
                // nicht verwendbar, der Sinn eines VAE ist ja gerade nach der Wahrscheinlichkeit
                // der Rekonstruktion zu fragen und nicht nach dem Rekontrstruktionsfehler
                // Exception in thread "main" java.lang.IllegalStateException: 
                // Cannot use reconstructionError method unless the variational 
                // autoencoder is configured with a standard loss function 
                // (via LossFunctionWrapper). For VAEs utilizing a reconstruction distribution, 
                // use the reconstructionProbability or reconstructionLogProbability methods (layer name: layer1, layer index: 1)
                //INDArray prediction = vae.reconstructionError(Nd4j.create(step)); 

                // Argument ist vermutlich die Aktivierung oder der input des VAE layers
                // retuns a column vector of reconstruction log probabilities for each example (shape: [numExamples,1])
                INDArray vaeIn = vae.input();
                INDArray prob = vae.reconstructionLogProbability(vaeIn, 1);
                // da darf ich nicht step reinstecken!
                //INDArray prob = vae.reconstructionLogProbability(Nd4j.create(step), 1);
                //result[0][f] = prob.getDouble(0);
                
                // ob das mit mehreren Dimensionen wirklich auch geht ist noch zu testen
                //FIXME
                for (int i=0;i<timeSerieWindow.length;i++){
                    // https://github.com/IsaacChanghau/StockPrediction/blob/master/src/main/java/com/isaac/stock/predict/StockPricePrediction.java
                    // getDouble(index des letzten Werts der Zeitreihe?)
                    // prediction
                    result[i][f] = prediction.getDouble(i); 
                    
                    // reconstruction propability
                    // lauter negative Werte streuen um -110
                    //result[i][f] =  prob.getDouble(i);
                }
            }
        }
        return result;
   }
   
   /**
    * @param timeSerieWindow [dims][frames]
    * @return score[dims][frames]
    */
   private double[][] scoreTimeSerieWindowComplete(double[][] window){
        double[][] result = new double[window.length][window[0].length];
        //Received input with size(1) = 560 (input array shape = [1, 560]); input.size(1) must match layer nIn size (nIn = 1)
        //INDArray prediction = net.output(Nd4j.create(timeSerieWindow));
        //WORKAROUND fixed dim=1
        int[] shape = new int[]{1, window.length, window[0].length};
        double[] flat = window[0];
        INDArray input = Nd4j.create(flat,shape,'c');
        INDArray resultINDArray =  net.output(input);
        for (int i=0;i<window[0].length;i++){
            result[0][i] = resultINDArray.getDouble(i);
        }
        return result;
    }
   
   protected static MultiLayerNetwork loadModel(){
       try {
           //log.info("Load model...");
           return ModelSerializer.restoreMultiLayerNetwork(new File("anomalydetection.zip"));
       } catch (IOException ex) {
           Logger.getLogger(AnomalyDetectorTest.class.getName()).log(Level.SEVERE, null, ex);
       }
       return null;
   }
}
