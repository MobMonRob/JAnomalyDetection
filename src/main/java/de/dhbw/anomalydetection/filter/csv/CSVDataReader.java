package de.dhbw.anomalydetection.filter.csv;

import de.dhbw.anomalydetection.filter.NumberedFileInputSplitExt;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import java.io.File;

import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.SequenceRecord;

import java.util.List;
import org.datavec.api.writable.Writable;

/**
 * @author Oliver Rettig
 */
public class CSVDataReader {
    
    private final SequenceRecordReader rr;
    private RecordMetaData currentRecordMetaData;
    
    //private final SequenceRecordReaderDataSetIterator dataSetIterator;
    
    private SequenceRecord currentRecord;
    private int currentRowIndex; // 
    private int currentFileRows; // Zahl der Zeilen im aktuellen File
    private int currentFileIndex = -1;
    
    private int maxLength;
    
    /**
     * @param maxLength only maxLength frames are read from each file
     * @throws java.io.FileNotFoundException
     * @throws java.lang.InterruptedException
     */
    public CSVDataReader(int maxLength) throws FileNotFoundException, IOException, InterruptedException, IllegalArgumentException {
        this();
        if (maxLength <=0){
            throw new IllegalArgumentException("maxlength = "+String.valueOf(maxLength)+" not allowed! maxLength must be > 0!");
        }
        this.maxLength = maxLength;
    }
   
    /*SequenceRecordReader getSequenceRecordReader(){
        return rr;
    }*/
    
    /**
     * All frames are read from the file.
     * 
     * @throws FileNotFoundException
     * @throws IOException
     * @throws InterruptedException 
     */
    public CSVDataReader() throws FileNotFoundException, IOException, InterruptedException {
        maxLength = -1;
        ClassPathResource cpr = new ClassPathResource("csv/71124z20.csv");
        File file = cpr.getFile();
        File folder = file.getParentFile();
        String name = file.getName().replace("20","%d");
        File f = new File(folder, name); //cpr.getFile().getAbsolutePath().replace("20", "%d");
        String path = f.getAbsolutePath();

        // https://deeplearning4j.org/datavecdoc/org/datavec/api/split/NumberedFileInputSplit.html
        // Es müssen alle Nummern durchgehend vorhanden sein in NumberedFileInputSplit
        // ausserdem ist nur "4" statt "04" möglich
        InputSplit is = new NumberedFileInputSplitExt(path, 10, 40); // 12, 24 zusammenhängender Bereich

        // tausender-Trennzeichen "," nicht erlaubt
        rr = new CSVSequenceRecordReader(1, ";");
        rr.initialize(is);
    }
    
    public String getFileName(){
        // oder irgendwie einen Listener einhängen um den aktuellen Filename zu beschaffen?
        if (currentRecordMetaData !=  null){
            return currentRecordMetaData.getLocation();
        } else {
            return "unknown file";
        }
    }
    public int getDimension(){
        //return getFieldSelection().getNumFields();
        return 1;
    }
    public int getCurrentFileRows(){
        return currentFileRows;
    }
    
    public boolean hasNext(){
       // bevor der erste record gelesen wurde, ist die Bendigung damit nicht erfüllt
       if (currentRowIndex < currentFileRows-1) return true;
       
       return rr.hasNext();
       //return dataSetIterator.hasNext();
    }
    /**
     * Get next frame with all available columns.
     * 
     * @return array with dim columns, typically dim==1.
     */
    public double[] next(){
        // bevor der erste record gelesen wurde, ist die Bedingung erfüllt
        // und dann immer beim letzten frame eines trials
        if (currentFileRows == currentRowIndex){
            currentRecord = rr.nextSequence(); 
            if (maxLength > 0){
                currentFileRows = maxLength;
            } else {
                currentFileRows = currentRecord.getSequenceLength();
            }
            currentRowIndex = 0;
            currentRecordMetaData = currentRecord.getMetaData();
            currentFileIndex++;
        }
        
      
        //get the dataset using the record reader. The datasetiterator handles vectorization
        // vermutlich bekomme ich mit jedem next() gleich die ganze Zeitreihe, das heißt
        // hasNext()/next() iterieret über die Files und nicht über die frames, jede row ist also eine Zeitreihe
        //DataSet dataSet = dataSetIterator.next();
        //String features = dataSet.getFeatures().getRow(0).toString();
        //INDArray res = dataSet.getFeatures();
        
        List<List<Writable>> data = currentRecord.getSequenceRecord();
        List<Writable> columns = data.get(currentRowIndex++);
        double[] result = new double[columns.size()];
        for (int i=0;i<columns.size();i++){
            result[i] = columns.get(i).toDouble();
        }
        return result;
    }
}
