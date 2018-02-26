package de.dhbw.anomalydetection.util;

import java.util.PrimitiveIterator.OfInt;
import java.util.Random;

/**
 *
 * @author Oliver Rettig
 */
public class RandomSequenceIndexGenerator {
    
    private final java.util.PrimitiveIterator.OfInt it;
    private final int windowSize;
    
    // PrimitiveIterator.OfInt randomIterator = new Random().ints(0, TRIAL_WINDOW_SIZE-SLIDING_WINDOW_SIZE).iterator();
    public RandomSequenceIndexGenerator(int windowSize){
      it = new Random().ints(0, windowSize).iterator();
      this.windowSize = windowSize;
    }
    
    public int[] getNextSequenz(){
         boolean[] used = new boolean[windowSize];
         for (int i=0;i<windowSize;i++){
             used[i] = false;
         }
         int[] result = new int[windowSize];
         
         int index = 0;
         while (index < windowSize){
             int test = it.nextInt();
             if (!used[test]){
                 result[index] = test;
                 used[test] = true;
                 index++;
             }
         }
         return result;
    }
}
