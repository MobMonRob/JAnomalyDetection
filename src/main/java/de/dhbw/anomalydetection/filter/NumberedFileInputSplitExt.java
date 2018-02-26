package de.dhbw.anomalydetection.filter;

import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.NoSuchElementException;

import java.util.List;
import java.util.ArrayList;

import org.datavec.api.split.NumberedFileInputSplit;

public class NumberedFileInputSplitExt extends NumberedFileInputSplit {
   
    private List<URI> uris = new ArrayList<URI>();
    
    /**
     * @param baseString String that defines file format. Must contain "%d", which will be replaced with
     *                   the index of the file, possibly zero-padded to x digits if the pattern is in the form %0xd.
     * @param minIdxInclusive Minimum index/number (starting number in sequence of files, inclusive)
     * @param maxIdxInclusive Maximum index/number (last number in sequence of files, inclusive)
     *                        @see {NumberedFileInputSplitTest}
     */
    public NumberedFileInputSplitExt(String baseString, int minIdxInclusive, int maxIdxInclusive) {
        super(baseString, minIdxInclusive, maxIdxInclusive);
        for (int i= minIdxInclusive; i<= maxIdxInclusive; i++){
            Path path = Paths.get(String.format(baseString, i));
            if (path.toFile().exists()){
                uris.add(path.toUri());
            }
        }
    }

    @Override
    public long length() {
        return uris.size();
    }

    @Override
    public URI[] locations() {
        return (URI[]) uris.toArray();
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        return new NumberedFileIteratorExt();
    }

    private class NumberedFileIteratorExt implements Iterator<String> {

        private int currIdx;

        private NumberedFileIteratorExt() {
            currIdx = 0;
        }

        @Override
        public boolean hasNext() {
            return currIdx <= length()-1;
        }

        @Override
        public String next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return uris.get(currIdx++).getPath();
            //return String.format(baseString, currIdx++);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }
}