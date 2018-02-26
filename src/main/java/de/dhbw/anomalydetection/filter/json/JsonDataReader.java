package de.dhbw.anomalydetection.filter.json;

import java.io.FileNotFoundException;
import java.io.IOException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.Record;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.awt.List;
import java.io.File;


import org.datavec.api.records.metadata.RecordMetaData;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * @author Rettig
 */
public class JsonDataReader {
    
    private final RecordReader rr;
    private RecordMetaData currentRecordMetaData;
    
    private final DataSetIterator dataSetIterator;
    
    public JsonDataReader(int windowSize) throws FileNotFoundException, IOException, InterruptedException {
        
        ClassPathResource cpr = new ClassPathResource("json/71124z20.json");
        File file = cpr.getFile();
        File folder = file.getParentFile();
        String name = file.getName().replace("20","%d");
        File f = new File(folder, name); //cpr.getFile().getAbsolutePath().replace("20", "%d");
        String path = f.getAbsolutePath();

        // https://deeplearning4j.org/datavecdoc/org/datavec/api/split/NumberedFileInputSplit.html
        InputSplit is = new NumberedFileInputSplit(path, 20, 22);

        rr = new JacksonLineRecordReader(getFieldSelection(), new ObjectMapper(new JsonFactory()));
        rr.initialize(is);
        
        //DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, 128)
        //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
        // treated as features. Columns indexes start at 0
        //.regression(labelColFrom, labelColTo)
        //.build();
        
        
        // https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExample.java
        
        //reader,label index,number of possible labels
        dataSetIterator = new RecordReaderDataSetIterator(rr,0,0,1);
    }
    
    public String getFileName(){
        //TODO
        // oder irgendwie einen Listener einhängen um den aktuellen Filename zu beschaffen
        
        if (currentRecordMetaData !=  null){
            return currentRecordMetaData.getLocation();
        } else {
            return "unknown file";
        }
    }
    public int getDimension(){
        return 3; // entsprechend field selection
    }
    /*private static FieldSelection getFieldSelection() {
        return new FieldSelection.Builder().addField("a").addField(new Text("MISSING_B"), "b")
                        .addField(new Text("MISSING_CX"), "c", "x").build();
    }*/
    
    private static FieldSelection getFieldSelection() {
        return new FieldSelection.Builder().
        		addField("T").
        		addField("ddHeight").
                        build();
    }
    
    public boolean hasNext(){
       //return rr.hasNext();
       return dataSetIterator.hasNext();
    }
    public double[] next(){
        //TODO
        // Konversion in ein Frame-Objekt
        //Record record = rr.nextRecord(); //next();
        //rr.
        
        //List<Writable> json = rr.next();
        //currentRecordMetaData = record.getMetaData();
        
        //get the dataset using the record reader. The datasetiterator handles vectorization
        //DataSet dataSet = dataSetIterator.next();
        
        return null;
    }
}
