package de.dhbw.anomalydetection;

import de.dhbw.anomalydetection.filter.csv.CSVDataReader;
import de.dhbw.anomalydetection.util.MathUtils;
import de.dhbw.anomalydetection.util.RandomSequenceIndexGenerator;
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

import java.io.FileNotFoundException;
import java.io.IOException;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
        
import org.deeplearning4j.ui.api.UIServer;

import java.io.File;


import org.nd4j.linalg.util.ArrayUtil;


import org.deeplearning4j.nn.conf.BackpropType;
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
public class AnomalyDetectorTrain {
    
   // Random number generator seed, for reproducability
   private static final int seed = 12345;
    
   // Network learning rate
   // The learning rate, or step rate, is the rate at which a function steps 
   // through the search space. The typical value of the learning rate is between 
   // 0.001 and 0.1. Smaller steps mean longer training times, but can lead to 
   // more precise results.
   private static final double learningRate = 0.2; // 0.01 bei keras  // 0.2 hat bei VAE funktioniert
        
   private static final int EPOCHS = 5;
   private static final int SLIDING_WINDOW_SIZE = 100;
   private static final int MINI_BATCH_SIZE = 50;
   
   // bei Silvan 700 als kurz nach dem Überfahren des ersten Hubbels
   private static final int TRIAL_WINDOW_SIZE = 350; //560; //350; //560; //35; //70; //140; //280; // 560 ist alles, bis 60% ohne anomalie
   //private static final long MINIMUM_COUNT_OF_TRAINING_WINDOWS = 23; //212; //106; //52; //26; // count of training frames before start to score
   
   private static long windowsTrained; // count of trained windows
   private static int frameIndex = 0; // processed frame index inside a window
   
   private static int currentTrial; 
   
   private static int dim; // dimensionality of the input data, defined by the JsonDataMovingWindow     
      
   // Zahl der nodes an den Enden des variational autoencoders
   private static final int RANGE_NODES_COUNT = SLIDING_WINDOW_SIZE; // 512; // ursprünglich 10
   
   private static final double DROPOUT_VALUE = 0.2d;
   
   private static MultiLayerNetwork net;
   private static AnomalyDetectorTrain detector;
      
   private final Random rng = new Random(seed);
   
   private org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae;
   
   public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException {
       for (int epoch = 0;epoch < EPOCHS;epoch++){ 
            //net.rnnClearPreviousState(); // clear previous state
            train(epoch);
            windowsTrained = 0 ;
       }
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
       MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder() // MultiLayerNetwork entspricht Keras model = Sequential()
        .seed(12345)
        // Generally, you want to use multiple epochs and one iteration (.iterations(1) option) 
        // when training; multiple iterations are generally only used when doing 
        // full-batch training on very small data sets.        
        .iterations(1)
        .weightInit(WeightInit.XAVIER) // vergleich mit Silvans code ZERO
        .updater(Updater.ADAGRAD) // for RMSProp in keras
        .activation(Activation.TANH)
        //.activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT/*STOCHASTIC_GRADIENT_DESCENT*/) // rmsprop bei keras
        .learningRate(learningRate)

        .regularization(false) // false bei keras
        // Lamda regularisation constant as discussed here:
        // http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
        //.l2(0.0001)
        .miniBatch(true) // neu: unklar ob es das bringt, da sample size klein
        .list()
               
        //.gateActivationFunction(Activation.HARDSIGMOID)
        // in keras entspricht das recurrent_activation
        .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(channels).nOut(64).gateActivationFunction(Activation.HARDSIGMOID)
          .dropOut(DROPOUT_VALUE).build())
        .layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(64).nOut(256).gateActivationFunction(Activation.HARDSIGMOID)
          .dropOut(DROPOUT_VALUE).build())
        .layer(2, new GravesLSTM.Builder().activation(Activation.TANH).nIn(256).nOut(100).gateActivationFunction(Activation.HARDSIGMOID)
          .dropOut(DROPOUT_VALUE).build())
               
        //.layer(3, new DenseLayer.Builder().nIn(100).nOut(100).activation(Activation.RELU).build())
                 
        // Output layer ist bereits ein DenseLayer
        // score and error calculation (of prediction vs. actual), given a loss function etc. 
        // The RnnOutputLayer layer type does not have any internal state, as it 
        // does not have any recurrent connections.
        .layer(3, new RnnOutputLayer.Builder() // MSE in keras
          .activation(Activation.IDENTITY).nIn(100).nOut(channels).lossFunction(LossFunctions.LossFunction.MSE).build())
               
        // seems to be mandatory
        // according to agibsonccc: You typically only use that with
        // pretrain(true) when you want to do pretrain/finetune without changing
        // the previous layers finetuned weights that's for autoencoders and
        // rbms
        // folgendes hier gefunden: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/variational/VariationalAutoEncoderExample.java
        // layerwise pretraining not supported from LSTM
        .pretrain(false).backprop(true)
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTForwardLength(100)
        .tBPTTBackwardLength(100).build();
       
        net = new MultiLayerNetwork(conf);
        net.init();
   }
   
   private void buildAutoencoder2(int channels){
       MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder() // MultiLayerNetwork entspricht Keras model = Sequential()
        .seed(12345)
        // Generally, you want to use multiple epochs and one iteration (.iterations(1) option) 
        // when training; multiple iterations are generally only used when doing 
        // full-batch training on very small data sets.        
        .iterations(1)
        .weightInit(WeightInit.ZERO) // vergleich mit Silvans code ZERO
        .updater(Updater.RMSPROP) // for RMSProp in keras
        //.activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT/*STOCHASTIC_GRADIENT_DESCENT*/) // rmsprop bei keras
        .learningRate(learningRate)

        .regularization(false) // false bei keras
        // Lamda regularisation constant as discussed here:
        // http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
        //.l2(0.0001)
        .miniBatch(true) // neu: unklar ob es das bringt, da sample size klein
        .list()
               
        //.gateActivationFunction(Activation.HARDSIGMOID)
        // in keras entspricht das recurrent_activation
        .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(channels).nOut(64).gateActivationFunction(Activation.HARDSIGMOID)
          .dropOut(DROPOUT_VALUE).build())
        .layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(64).nOut(256).gateActivationFunction(Activation.HARDSIGMOID)
          .dropOut(DROPOUT_VALUE).build())
        .layer(2, new GravesLSTM.Builder().activation(Activation.TANH).nIn(256).nOut(100).gateActivationFunction(Activation.HARDSIGMOID)
          .dropOut(DROPOUT_VALUE).build())
               
        //.layer(3, new DenseLayer.Builder().nIn(100).nOut(100).activation(Activation.RELU).build())
                 
        // Output layer ist bereits ein DenseLayer
        // score and error calculation (of prediction vs. actual), given a loss function etc. 
        // The RnnOutputLayer layer type does not have any internal state, as it 
        // does not have any recurrent connections.
        .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE) // MSE in keras
          .activation(Activation.IDENTITY).nIn(100).nOut(channels).build())
               
        // seems to be mandatory
        // according to agibsonccc: You typically only use that with
        // pretrain(true) when you want to do pretrain/finetune without changing
        // the previous layers finetuned weights that's for autoencoders and
        // rbms
        // folgendes hier gefunden: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised/variational/VariationalAutoEncoderExample.java
        // layerwise pretraining not supported from LSTM
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
   private AnomalyDetectorTrain(int channels){
       
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
    * Read TRIAL_WINDOW_SIZE frames from each file. Move a window of size SLIDING_WINDOW_SIZE 
    * over the data and save the frames for each step.
    * 
    * @param epoch
    * @throws FileNotFoundException
    * @throws IOException 
    * @throws java.lang.InterruptedException 
    */
   public static void train(int epoch) throws FileNotFoundException, IOException, InterruptedException {
        CSVDataReader data = new CSVDataReader(TRIAL_WINDOW_SIZE); 
            
        if (detector == null){
            dim = data.getDimension();
            detector = new AnomalyDetectorTrain(dim); 
        }
        // vielleicht besser durch einen Circularbuffer ersetzen?
        double[][] window = new double[dim][TRIAL_WINDOW_SIZE];

        currentTrial = 0;
        
        // über die frames der Trainingsdaten (aller input Dateien) iterieren
        while (data.hasNext()){
            double[] frame = data.next();

            if (frameIndex < TRIAL_WINDOW_SIZE){
                //System.arraycopy(frame, 0, window[frameIndex], 0, dim);
                for (int i=0;i<dim;i++){
                    window[i][frameIndex] = frame[i];
                }
                frameIndex++;
            } else {
                // 1. count of sliding windows, 2. dims, 3. sliding window size
                double[][][] multidimwindow = new double[TRIAL_WINDOW_SIZE-SLIDING_WINDOW_SIZE][dim][SLIDING_WINDOW_SIZE];
                
                // sliding windows durchiterieren
                for (int w=0;w<TRIAL_WINDOW_SIZE-SLIDING_WINDOW_SIZE;w++){
                    // dimensions durchiterieren
                    for (int i=0;i<dim;i++){
                        // frames eines sliding windows durchiterieren
                        for (int f=0;f<SLIDING_WINDOW_SIZE;f++){
                            multidimwindow[w][i][f] = window[i][w+f];
                        }
                    }
                }
                
                
                // mini batch size berücksichtigen
                // windows durcheinanderwürfeln
                
                RandomSequenceIndexGenerator r = new RandomSequenceIndexGenerator(MINI_BATCH_SIZE);
                
                // batches durchiterieren
                int completeBatches = (TRIAL_WINDOW_SIZE-SLIDING_WINDOW_SIZE)/MINI_BATCH_SIZE;
                System.out.println("Complete batches "+String.valueOf(completeBatches));
                for (int i=0; i < completeBatches; i++){
                    
                    // windows eines batches durchiterieren
                    double[][][] randomData = new double[MINI_BATCH_SIZE][dim][SLIDING_WINDOW_SIZE];
                    int[] randomIndizes = r.getNextSequence();
                    for (int index=0;index < MINI_BATCH_SIZE; index++){
                        int randomIndex = randomIndizes[index];
                        //System.out.println("index="+String.valueOf(MINI_BATCH_SIZE*index+randomIndex));
                        for (int d=0;d<dim;d++){
                            for (int f=0;f<SLIDING_WINDOW_SIZE;f++){
                                randomData[index][d][f] = multidimwindow[i*MINI_BATCH_SIZE+randomIndex][d][f];
                            }
                        }
                    }
                    
                    double score = detector.train(randomData, data.getFileName());
                    System.out.println(data.getFileName()+": epoch = "+String.valueOf(epoch)+"("+String.valueOf(EPOCHS)+") batch "+i+" score = "+String.valueOf(score)+"!");
                }
                
                // die restlichen trial windows zu einem batch zusammenfassen
                // TODO Reihenfolge hier auch noch durchwürfeln
                int REST_MINI_BATCH_SIZE = TRIAL_WINDOW_SIZE-SLIDING_WINDOW_SIZE // Zahl der Windows
                        - (completeBatches * MINI_BATCH_SIZE);
                if (REST_MINI_BATCH_SIZE > 0){
                    int[] randomIndizes = (new RandomSequenceIndexGenerator(REST_MINI_BATCH_SIZE)).getNextSequence();
                    double[][][] randomData = new double[REST_MINI_BATCH_SIZE][dim][SLIDING_WINDOW_SIZE];
                    for (int index = 0; index < REST_MINI_BATCH_SIZE;index++){
                        int randomIndex = randomIndizes[index];
                        for (int d=0;d<dim;d++){
                            for (int f=0;f<SLIDING_WINDOW_SIZE;f++){
                                randomData[index][d][f] = multidimwindow[completeBatches*MINI_BATCH_SIZE + randomIndex][d][f];
                            }
                        }
                    }
                    double score = detector.train(randomData, data.getFileName());
                    System.out.println(data.getFileName()+": epoch= "+String.valueOf(epoch)+"("+String.valueOf(EPOCHS)+") batch "+completeBatches+" score = "+String.valueOf(score)+"!");
                }
                frameIndex = 0;
                currentTrial++;
            }
        }
        frameIndex = 0;
        currentTrial = 0;
        saveModel(new File("anomalydetection.zip"), net);
        //ModelSerializer.writeModel(net, new File("anomalydetection.zip"), false);
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
  
   
  
    /**
    * Train the net.
    *
    * @param window [min batch size][dim][windowsize]
    * Jede Zeile hat so viele Werte wie die Fensterbreite festlegt, es könnte 
    * 3 Spalten geben, je eine für eine der 3 Dimensionen eines Beschleunigungssensors, 
    * ausserdem gibt es eine dimension für die Zahl der examples.
    * @param fileName
    * @return 
    * 
    */
   public double train(double[][][] window, String fileName){
      MathUtils.normalize(window);
      double[] flat = ArrayUtil.flattenDoubleArray(window);
      int[] shape = new int[]{window.length, window[0].length, window[0][0].length};	// 50, 1, 100
      INDArray input = Nd4j.create(flat,shape,'c');
      
      // inputdata, expected output values, or labels
      // eigentlich müssten die labels um einen frame verschoben werden
      // aber vermutlich macht das keinen Unterschied.
      net.fit(input, input);
      windowsTrained++;
      return net.score();
      // funktioniert nicht, vermutlich, da ich hier keinen batch übergeben darf
      // return net.f1Score(input,input);
    }
   
   
    protected static void saveModel(File file, MultiLayerNetwork net) throws IOException {
        // log.info("Saving model...");
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, file, false);
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
