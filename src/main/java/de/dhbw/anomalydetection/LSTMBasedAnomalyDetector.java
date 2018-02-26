package de.dhbw.anomalydetection;

import de.dhbw.anomalydetection.util.PlotUtils;
import de.dhbw.anomalydetection.filter.CSVDataReader;
import java.util.Random;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
//import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
        
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;

import java.util.List;
import java.util.ArrayList;
import org.jfree.data.xy.XYSeries;

import org.nd4j.linalg.util.ArrayUtil;

import org.apache.commons.math3.stat.StatUtils;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;

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
public class LSTMBasedAnomalyDetector {
    
   // Random number generator seed, for reproducability
   private static final int seed = 12345;
    
   // Network learning rate
   // The learning rate, or step rate, is the rate at which a function steps 
   // through the search space. The typical value of the learning rate is between 
   // 0.001 and 0.1. Smaller steps mean longer training times, but can lead to 
   // more precise results.
   private static final double learningRate = 0.2; // ehemals 0.01 bei keras 0.2 da %
        
   // bei Silvan 600
   private static final int WINDOW_SIZE = 340; //560; //350; //560; //35; //70; //140; //280; // 560 ist alles, bis 60% ohne anomalie
   private static final int TEST_WINDOW_SIZE = 550; 
   private static final long MINIMUM_COUNT_OF_TRAINING_WINDOWS = 23; //212; //106; //52; //26; // count of training frames before start to score
   
   private static long windowsTrained; // count of trained windows
   private static int frameIndex = 0; // processed frame index inside a window
   
   private static int currentWindow; // the index of the current processed window in a file
   
   private static int dim; // dimensionality of the input data, defined by the JsonDataMovingWindow     
      
   // Zahl der nodes an den Enden des variational autoencoders
   private static final int RANGE_NODES_COUNT = 512; // ursprünglich 10
   
   private static MultiLayerNetwork net;
   private static LSTMBasedAnomalyDetector detector;
      
   private final Random rng = new Random(seed);
   
   private org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae;
   
   public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException {
       for (int epoch = 0;epoch < 50;epoch++){ // 80
            //net.rnnClearPreviousState(); // clear previous state
            train();
            windowsTrained = 0 ;
       }
       singleStepsTest();
       while(true);
   }
   
   private void buildVAENet(int channels){
       MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(12345)
                
        .iterations(1)
        // You need to make sure your weights are neither too big nor too small. 
        // Xavier weight initialization is usually a good choice for this
        .weightInit(WeightInit.XAVIER)
        .updater(Updater.ADAGRAD)
        .activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(learningRate)

        .regularization(true)
        // Lamda regularisation constant as discussed here:
        // http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
        .l2(0.0001)

        .list()
        //  A long-short term memory (LSTM) layer responsible for recognizing temporal 
        // patterns in our time-series sensor data stream.
        // Supervised Sequence Labelling with Recurrent Neural Networks http://www.cs.toronto.edu/~graves/phd.pdf
        .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(channels).nOut(RANGE_NODES_COUNT)
          .build())

        // To detect anomalies use an autoencoder
        .layer(1, new VariationalAutoencoder.Builder()
          .activation(Activation.LEAKYRELU)
          .encoderLayerSizes(256, 256) //2 encoder layers, each of size 256
          .decoderLayerSizes(256, 256) //2 decoder layers, each of size 256
          .pzxActivationFunction(Activation.IDENTITY) //p(z|data) activation function
                // https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/nn/layers/variational/TestVAE.java
          //Bernoulli reconstruction distribution + sigmoid activation - for modelling 
          //binary data (or data in range 0 to 1)
          //.reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
          // Da Daten nicht binär: Multivariate Gaussian 
          .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.IDENTITY))
          //.reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))  
          .nIn(RANGE_NODES_COUNT) //Input size: 28x28
          .nOut(RANGE_NODES_COUNT) //Size of the latent variable space: p(z|x) - 32 values
          .build())

        // Output layer
        // score and error calculation (of prediction vs. actual), given a loss function etc. 
        // The RnnOutputLayer layer type does not have any internal state, as it 
        // does not have any recurrent connections.
        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
          .activation(Activation.IDENTITY).nIn(RANGE_NODES_COUNT).nOut(channels).build())
        .pretrain(false).backprop(true)
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTForwardLength(100)
        .tBPTTBackwardLength(100).build();
        //TODO
        // folgendes hier gefunden: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/variational/VariationalAutoEncoderExample.java
        // .pretrain(true).backprop(true).build();
        net = new MultiLayerNetwork(conf);
        net.init();
        //Get the variational autoencoder layer:
        vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(1);
   }
   
   private void buildAutoencoder(int channels){
       MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(12345)
        // Generally, you want to use multiple epochs and one iteration (.iterations(1) option) 
        // when training; multiple iterations are generally only used when doing 
        // full-batch training on very small data sets.        
        .iterations(1)
        .weightInit(WeightInit.ZERO) // vergleich mit Silvans code ZERO
        .updater(Updater.RMSPROP) // for RMSProp
        //.activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // rmsprop bei keras
        .learningRate(learningRate)

        .regularization(false) // false bei keras
        // Lamda regularisation constant as discussed here:
        // http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
        .l2(0.0001)
        .miniBatch(false) // neu: unklar ob es das bringt, da sample size klein
        .list()
               
               //.gateActivationFunction(Activation.HARDSIGMOID)
               // scheint nicht zu funktionieren-->score wird NaN
        .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(channels).nOut(64)//.gateActivationFunction(Activation.HARDSIGMOID)
          .build())
        .layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(64).nOut(256)//.gateActivationFunction(Activation.HARDSIGMOID)
          .build())
        .layer(2, new GravesLSTM.Builder().activation(Activation.TANH).nIn(256).nOut(100)//.gateActivationFunction(Activation.HARDSIGMOID)
          .build())
        //.layer(3, new DenseLayer.Builder().nIn(100).nOut(100).activation(Activation.RELU).build())
                 
        // Output layer ist bereits ein DenseLayer
        // score and error calculation (of prediction vs. actual), given a loss function etc. 
        // The RnnOutputLayer layer type does not have any internal state, as it 
        // does not have any recurrent connections.
        .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
          .activation(Activation.IDENTITY).nIn(100).nOut(channels).build())
               
        // seems to be mandatory
        // according to agibsonccc: You typically only use that with
        // pretrain(true) when you want to do pretrain/finetune without changing
        // the previous layers finetuned weights that's for autoencoders and
        // rbms
        // folgendes hier gefunden: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/variational/VariationalAutoEncoderExample.java
        .pretrain(false).backprop(true)
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTForwardLength(100)
        .tBPTTBackwardLength(100).build();
       
        net = new MultiLayerNetwork(conf);
        net.init();
   }
   
   /**
    * Constructor.
    * 
    * @param channels dimensionality of the data
    */
   private LSTMBasedAnomalyDetector(int channels){
       
        //buildVAENet(channels);
        buildAutoencoder(channels);
        
        // add an listener which outputs the error every 1 parameter updates
        //net.setListeners(new ScoreIterationListener(1));   
        
        //net.printConfiguration();
        
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        net.setListeners(new StatsListener(statsStorage));
        net.init(); // testweise da mir scheint, dass sonst die listener nicht eingehängt sind
        
        //net.printConfiguration();
        
        //Finally: open your browser and go to http://localhost:9000/train
        
       
   }
   
   /**
    * Process data files with collected data.
    * 
    * @throws FileNotFoundException
    * @throws IOException 
    * @throws java.lang.InterruptedException 
    */
   public static void train() throws FileNotFoundException, IOException, InterruptedException {
        CSVDataReader data = new CSVDataReader(WINDOW_SIZE); 
            
        if (detector == null){
            dim = data.getDimension();
            detector = new LSTMBasedAnomalyDetector(dim); 
        }
        // vielleicht besser durch einen Circularbuffer ersetzen?
        double[][] window = new double[dim][WINDOW_SIZE];

        currentWindow = 0;
        
        // über die frames einer Datei iterieren
        while (data.hasNext()){
            double[] frame = data.next();

            if (frameIndex < WINDOW_SIZE){
                //System.arraycopy(frame, 0, window[frameIndex], 0, dim);
                for (int i=0;i<dim;i++){
                    window[i][frameIndex] = frame[i];
                }
                frameIndex++;
            } else {
                // das ist alles nur nötig, wenn ich fft auf einem fenster machen will
                // sonst könnte ich auch einfach window[][] statt multdimwindow[][][] übergeben
                double[][][] multidimwindow = new double[1][dim][WINDOW_SIZE];
                for (int i=0;i<dim;i++){
                    double[] array = window[i];
                    //double[] fftWindow = fft(array);
                    for (int f=0;f<WINDOW_SIZE;f++){
                        // mit fft
                        //multidimwindow[i][f] = fftWindow[f];
                        // ohne fft
                        multidimwindow[0][i][f] = array[f];
                    }
                }
                
                // score für das komplette Zeitfenster
                double score = detector.applyWindow(multidimwindow, data.getFileName());
                System.out.println(data.getFileName()+": window "+currentWindow+
                        ", score/frame = "+String.valueOf(score/WINDOW_SIZE));
                frameIndex = 0;
                currentWindow++;
            }
        }
        frameIndex = 0;
        currentWindow = 0;
   }
   
   /**
    * Normalize by subtraction of the trials mean and division by std.
    * 
    * @param multidimwindow [1][dim][WINDOW_SIZE]
    * @return [1][dim][WINDOW_SIZE]
    */
   private void normalize(double[][][] multidimwindow){
       for (int d=0;d<multidimwindow[0].length;d++){
            multidimwindow[0][d] = StatUtils.normalize(multidimwindow[0][d]);
       }
   }
   
   /*DataNormalization train2(){
       DataSetIterator trainData = (new CSVDataSetReader()).getDataSetIterator();
       
       //Normalize the training data
       DataNormalization normalizer = new NormalizerStandardize();
       normalizer.fit(trainData);              //Collect training data statistics
       trainData.reset();

       //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
       trainData.setPreProcessor(normalizer);
       
       net.fit(trainData);
       
       return normalizer;
   }*/
   
   
   
   public static void singleStepsTest() throws FileNotFoundException, IOException, InterruptedException {
     
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
                double[][] result = detector.scoreTimeSerieWindow(window);
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
                /*for (int i=0;i<window[0].length;i++){
                    plot[1][i] *= max0/max1; 
                }*/
                PlotUtils.createChart(data.getFileName(), plot, new String[]{"orig", "score"});
                //System.out.println(data.getFileName()+": window "+currentWindow+", score = "+score);
                frameIndex = 0;
                currentWindow++;
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
   private double[][] scoreTimeSerieWindow(double[][] timeSerieWindow){
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
            // returns output activations for the current frame
            // output for the rnnTimeStep and the output/feedForward methods 
            // should be identical (for each time step), whether we make these 
            // predictions all at once (output/feedForward) or whether these 
            // predictions are generated one or more steps at a time (rnnTimeStep). 
            // Thus, the only difference should be the computational cost.
            // https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/encdec/EncoderDecoderLSTM.java
            INDArray values = net.rnnTimeStep(Nd4j.create(step));
            
            if (vae == null){
                // ob das mit mehreren Dimensionen wirklich auch geht ist noch zu testen
                //FIXME
                for (int i=0;i<timeSerieWindow.length;i++){
                    // https://github.com/IsaacChanghau/StockPrediction/blob/master/src/main/java/com/isaac/stock/predict/StockPricePrediction.java
                    // getDouble(index des letzten Werts der Zeitreihe?)
                    result[i][f] = values.getDouble(i); 
                }
            } else {
                // vae.score(); scheint immer 0 zu sein
                
                // nicht verwendbar, der Sinn eines VAE ist ja gerade nach der Wahrscheinlichkeit
                // der Rekonstruktion zu fragen und nicht nach dem Rekontrstruktionsfehler
                //INDArray values = vae.reconstructionError(Nd4j.create(step)); 

                //TODO genau das brauche ich 
                // Argument ist vermutlich die Aktivierung oder der input
                //values = vae.reconstructionLogProbability(..., 1);
                // da darf ich nicht step reinstecken, sondern vermutlich die Aktivierung am 
                // Eingang des VAEs
                //INDArray prob = vae.reconstructionLogProbability(Nd4j.create(step), 1);
                //result[0][f] = prob.getDouble(0);
            }
        }
        return result;
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
   
   /**
    * Train the net and if the net is trained additionlly determine a score for 
    * the complete time window.
    *
    * @param window [1][dim][windowsize]
    * Jede Zeile hat so viele Werte wie die Fensterbreite festlegt, es könnte 
    * 3 Spalten geben, je eine für eine der 3 Dimensionen eines Beschleunigungssensors, 
    * ausserdem gibt es eine dimension für die Zahl der examples, die aber bisher immer
    * eins ist.
    * @param fileName
    * 
    * @return -1, if the net is not yet trained enough, else the score
    */
   public double applyWindow(double[][][] window, String fileName){
      double result = -1;
      normalize(window);
      double[] flat = ArrayUtil.flattenDoubleArray(window);
      int[] shape = new int[]{window.length, window[0].length, window[0][0].length};	
      INDArray input = Nd4j.create(flat,shape,'c');
      
      if (windowsTrained < MINIMUM_COUNT_OF_TRAINING_WINDOWS){
            // inputdata, expected output values, or labels
            // eigentlich müssten die labels um einen frame verschoben werden
            // aber vermutlich macht das keinen Unterschied.
            net.fit(input, input);
            windowsTrained++;
            double[][] plotwindow = new double[1][window[0][0].length];
            for (int f=0;f<window[0][0].length;f++){
                plotwindow[0][f] = window[0][0][f];
            }
            //PlotUtils.createChart(fileName, plotwindow, new String[]{"orig"});
      } else {
            double[][] plotwindow = new double[3][window[0][0].length];
            for (int i=0;i<window[0][0].length;i++){
                plotwindow[0][i] = window[0][0][i];
            }
            // conventional autoencoder
            if (vae == null){
                /**
                  * Label the probabilities of the input
                  * input is the input to label
                  * returns a vector of probabilities given each label.
                  * This is typically of the form:
                  * [0.5, 0.5] or some other probability distribution summing to one
                  */
                // vermutlich falsch, wie gross ist überhaupt das array?, 
                //TODO wie bekomme ich die Aktivierung des output-layers
                INDArray output = net.output(input/*,true*/);
                
                // Compute activations from input to output of the output layer
                List<INDArray> predictedList = net.feedForward();
                INDArray predicted = predictedList.get(predictedList.size()-1);
                //Evaluation evaluation_validate = new Evaluation(2);
                //evaluation_validate.evalTimeSeries(input, predicted);
                //System.out.println(evaluation_validate.stats() ); 
                
                for (int i=0;i<window[0][0].length;i++){
                    plotwindow[1][i] = output.getDouble(i);
                    plotwindow[2][i] = predicted.getDouble(i);
                }
            // variational autoencoder
            } else {
                //INDArray latentSpaceValues = vae.activate(input, false);
                //INDArray out = vae.generateAtMeanGivenZ(latentSpaceGrid);
            }
            
            //TODO
            // pred-orig quadrieren und auf 1 normieren und als score plottten
            PlotUtils.createChart(fileName, plotwindow, new String[]{"orig","out","pred"});
            
            // für die ganze Zeitreihe, 
            // score (loss function values) of the prediction with respect to the true labels during training
            result = net.score(new DataSet(input,input), false);
      }
      return result;
    }
   
    public void saveModel(File file){
        //log.info("Saving model...");
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        //ModelSerializer.writeModel(net, file, true);
    }
    public void loadModel(File file){
        //log.info("Load model...");
        //net = ModelSerializer.restoreMultiLayerNetwork(file);
        //log.info("Testing...");
    }
    
    /**
     * Calculate the reconstruction probability, as described in An & Cho, 2015 - "Variational Autoencoder based
     * Anomaly Detection using Reconstruction Probability" (Algorithm 4)<br>
     * The authors describe it as follows: "This is essentially the probability of the data being generated from a given
     * latent variable drawn from the approximate posterior distribution."<br>
     * <br>
     * Specifically, for each example x in the input, calculate p(x). Note however that p(x) is a stochastic (Monte-Carlo)
     * estimate of the true p(x), based on the specified number of samples. More samples will produce a more accurate
     * (lower variance) estimate of the true p(x) for the current model parameters.<br>
     * <br>
     * Internally uses {@link #reconstructionLogProbability(INDArray, int)} for the actual implementation.
     * That method may be more numerically stable in some cases.<br>
     * <br>
     * The returned array is a column vector of reconstruction probabilities, for each example. Thus, reconstruction probabilities
     * can (and should, for efficiency) be calculated in a batched manner.
     *
     * @param data       The data to calculate the reconstruction probability for
     * @param numSamples Number of samples with which to base the reconstruction probability on.
     * @return Column vector of reconstruction probabilities for each example (shape: [numExamples,1])
     */
    /**
     * https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/variational/VariationalAutoencoder.java
     * 
     * Return the log reconstruction probability given the specified number of samples.<br>
     * See {@link #reconstructionLogProbability(INDArray, int)} for more details
     *
     * @param data       The data to calculate the log reconstruction probability
     * @param numSamples Number of samples with which to base the reconstruction probability on.
     * @return Column vector of reconstruction log probabilities for each example (shape: [numExamples,1])
     */
    /*public INDArray reconstructionLogProbability(INDArray data, int numSamples) {
        if (numSamples <= 0) {
            throw new IllegalArgumentException(
                            "Invalid input: numSamples must be > 0. Got: " + numSamples + " " + layerId());
        }
        if (reconstructionDistribution instanceof LossFunctionWrapper) {
            throw new UnsupportedOperationException("Cannot calculate reconstruction log probability when using "
                            + "a LossFunction (via LossFunctionWrapper) instead of a ReconstructionDistribution: ILossFunction "
                            + "instances are not in general probabilistic, hence it is not possible to calculate reconstruction probability "
                            + layerId());
        }

        //Forward pass through the encoder and mean for P(Z|X)
        setInput(data);
        VAEFwdHelper fwd = doForward(true, true);
        IActivation afn = layerConf().getActivationFn();

        //Forward pass through logStd^2 for P(Z|X)
        INDArray pzxLogStd2W = getParamWithNoise(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W, false);
        INDArray pzxLogStd2b = getParamWithNoise(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_B, false);

        INDArray meanZ = fwd.pzxMeanPreOut;
        INDArray logStdev2Z = fwd.encoderActivations[fwd.encoderActivations.length - 1].mmul(pzxLogStd2W)
                        .addiRowVector(pzxLogStd2b);
        pzxActivationFn.getActivation(meanZ, false);
        pzxActivationFn.getActivation(logStdev2Z, false);

        INDArray pzxSigma = Transforms.exp(logStdev2Z, false);
        Transforms.sqrt(pzxSigma, false);

        int minibatch = input.size(0);
        int size = fwd.pzxMeanPreOut.size(1);

        INDArray pxzw = getParamWithNoise(VariationalAutoencoderParamInitializer.PXZ_W, false);
        INDArray pxzb = getParamWithNoise(VariationalAutoencoderParamInitializer.PXZ_B, false);

        INDArray[] decoderWeights = new INDArray[decoderLayerSizes.length];
        INDArray[] decoderBiases = new INDArray[decoderLayerSizes.length];

        for (int i = 0; i < decoderLayerSizes.length; i++) {
            String wKey = "d" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "d" + i + BIAS_KEY_SUFFIX;
            decoderWeights[i] = getParamWithNoise(wKey, false);
            decoderBiases[i] = getParamWithNoise(bKey, false);
        }

        INDArray sumReconstructionNegLogProbability = null;
        for (int i = 0; i < numSamples; i++) {
            INDArray e = Nd4j.randn(minibatch, size);
            INDArray z = e.muli(pzxSigma).addi(meanZ); //z = mu + sigma * e, with e ~ N(0,1)

            //Do forward pass through decoder
            int nDecoderLayers = decoderLayerSizes.length;
            INDArray currentActivations = z;
            for (int j = 0; j < nDecoderLayers; j++) {
                currentActivations = currentActivations.mmul(decoderWeights[j]).addiRowVector(decoderBiases[j]);
                afn.getActivation(currentActivations, false);
            }

            //And calculate reconstruction distribution preOut
            INDArray pxzDistributionPreOut = currentActivations.mmul(pxzw).addiRowVector(pxzb);

            if (i == 0) {
                sumReconstructionNegLogProbability =
                                reconstructionDistribution.exampleNegLogProbability(data, pxzDistributionPreOut);
            } else {
                sumReconstructionNegLogProbability
                                .addi(reconstructionDistribution.exampleNegLogProbability(data, pxzDistributionPreOut));
            }
        }

        setInput(null);
        return sumReconstructionNegLogProbability.divi(-numSamples);
}
     */
}
