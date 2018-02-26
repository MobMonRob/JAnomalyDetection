package de.dhbw.anomalydetection.filter.csv.test;

import de.dhbw.anomalydetection.filter.NumberedFileInputSplitExt;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;

/**
 *
 * @author Oliver Rettig
 */
public class CSVDataSetReader {
    
    // regression - Whether output is for regression or classification
    private SequenceRecordReaderDataSetIterator dataSetIterator;
    
        // Beispiel für Sequence-data
        // https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIteratorTest.java
        
        // regression - Whether output is for regression or classification
        //dataSetIterator = new SequenceRecordReaderDataSetIterator(rr,rr1,1, 0, false, 
        //                                   SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START);
        
        // vermutlich für classification
        //MultiDataSetIterator dataSetIterator = new RecordReaderMultiDataSetIterator.Builder(1)
        //                .addSequenceReader("in", featureReader2).addSequenceReader("out", labelReader2).addInput("in")
        //.addOutputOneHot("out", 0, 4).build();
    
    private final SequenceRecordReader rr1;
    private final SequenceRecordReader rr2;
    
    public CSVDataSetReader() throws FileNotFoundException, IOException, InterruptedException, IllegalArgumentException {
         
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
        rr1 = new CSVSequenceRecordReader(1, ";");
        rr1.initialize(is);
        rr2 = new CSVSequenceRecordReader(1, ";");
        rr2.initialize(is);
        dataSetIterator = new SequenceRecordReaderDataSetIterator(rr1,rr2,1, 0, false, 
                                           SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START); 
        
        //dataSetIterator = new RecordReaderMultiDataSetIterator.Builder(1)
        //                .addSequenceReader("in", rr1).addSequenceReader("out", rr2).addInput("in")
        //                .addOutputOneHot("out", 0, 4).build();
    }
    
    public SequenceRecordReaderDataSetIterator getDataSetIterator(){
        return dataSetIterator;
    }
}
