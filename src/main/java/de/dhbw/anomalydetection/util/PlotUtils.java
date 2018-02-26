package de.dhbw.anomalydetection.util;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.AxisLocation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.GrayPaintScale;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.PaintScaleLegend;
import org.jfree.data.xy.*;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.RectangleInsets;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.*;

import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.chart.renderer.xy.XYSplineRenderer;

import java.util.List;
import java.util.ArrayList;
/**
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/PlotUtil.java
 * 
 * @author rettig
 */
public class PlotUtils extends ApplicationFrame {
    
    private static JFreeChart createChart(XYZDataset dataset, double[] mins, 
                                          double[] maxs, int nPoints, XYDataset xyData) {
        NumberAxis xAxis = new NumberAxis("X");
        xAxis.setRange(mins[0],maxs[0]);


        NumberAxis yAxis = new NumberAxis("Y");
        yAxis.setRange(mins[1], maxs[1]);

        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth((maxs[0]-mins[0])/(nPoints-1));
        renderer.setBlockHeight((maxs[1] - mins[1]) / (nPoints - 1));
        PaintScale scale = new GrayPaintScale(0, 1.0);
        renderer.setPaintScale(scale);
        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinesVisible(false);
        plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5));
        JFreeChart chart = new JFreeChart("", plot);
        chart.getXYPlot().getRenderer().setSeriesVisibleInLegend(0, false);


        NumberAxis scaleAxis = new NumberAxis("Probability (class 0)");
        scaleAxis.setAxisLinePaint(Color.white);
        scaleAxis.setTickMarkPaint(Color.white);
        scaleAxis.setTickLabelFont(new Font("Dialog", Font.PLAIN, 7));
        PaintScaleLegend legend = new PaintScaleLegend(new GrayPaintScale(),
                scaleAxis);
        legend.setStripOutlineVisible(false);
        legend.setSubdivisionCount(20);
        legend.setAxisLocation(AxisLocation.BOTTOM_OR_LEFT);
        legend.setAxisOffset(5.0);
        legend.setMargin(new RectangleInsets(5, 5, 5, 5));
        legend.setFrame(new BlockBorder(Color.red));
        legend.setPadding(new RectangleInsets(10, 10, 10, 10));
        legend.setStripWidth(10);
        legend.setPosition(RectangleEdge.LEFT);
        chart.addSubtitle(legend);

        ChartUtilities.applyCurrentTheme(chart);

        plot.setDataset(1, xyData);
        XYLineAndShapeRenderer renderer2 = new XYLineAndShapeRenderer();
        renderer2.setBaseLinesVisible(false);
        plot.setRenderer(1, renderer2);

        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);

        return chart;
    }
    
    private static JFreeChart createChart2(XYZDataset dataset, double[] mins, 
                                          double[] maxs, int nPoints, XYDataset xyData) {
        NumberAxis xAxis = new NumberAxis("X");
        xAxis.setRange(mins[0],maxs[0]);


        NumberAxis yAxis = new NumberAxis("Y");
        yAxis.setRange(mins[1], maxs[1]);

        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth((maxs[0]-mins[0])/(nPoints-1));
        renderer.setBlockHeight((maxs[1] - mins[1]) / (nPoints - 1));
        PaintScale scale = new GrayPaintScale(0, 1.0);
        renderer.setPaintScale(scale);
        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinesVisible(false);
        plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5));
        JFreeChart chart = new JFreeChart("", plot);
        chart.getXYPlot().getRenderer().setSeriesVisibleInLegend(0, false);


        NumberAxis scaleAxis = new NumberAxis("Probability (class 0)");
        scaleAxis.setAxisLinePaint(Color.white);
        scaleAxis.setTickMarkPaint(Color.white);
        scaleAxis.setTickLabelFont(new Font("Dialog", Font.PLAIN, 7));
        PaintScaleLegend legend = new PaintScaleLegend(new GrayPaintScale(),
                scaleAxis);
        legend.setStripOutlineVisible(false);
        legend.setSubdivisionCount(20);
        legend.setAxisLocation(AxisLocation.BOTTOM_OR_LEFT);
        legend.setAxisOffset(5.0);
        legend.setMargin(new RectangleInsets(5, 5, 5, 5));
        legend.setFrame(new BlockBorder(Color.red));
        legend.setPadding(new RectangleInsets(10, 10, 10, 10));
        legend.setStripWidth(10);
        legend.setPosition(RectangleEdge.LEFT);
        chart.addSubtitle(legend);

        ChartUtilities.applyCurrentTheme(chart);

        plot.setDataset(1, xyData);
        XYLineAndShapeRenderer renderer2 = new XYLineAndShapeRenderer();
        renderer2.setBaseLinesVisible(false);
        plot.setRenderer(1, renderer2);

        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);

        return chart;
    }
    
    /**
     * Create JFreeChart dataset.
     * 
     * @param multidimwindow [dim][windowsize]
     * @return dataSet
     */
    private static XYSeriesCollection createJFreeData(double[][] multidimwindow, String[] labels) {
        if (labels.length != multidimwindow.length) throw new IllegalArgumentException("Count of labels \""+
                String.valueOf(labels.length)+"\" does not match multidimwindow dim = \""+multidimwindow.length+"!");
        XYSeriesCollection result = new XYSeriesCollection();
        for (int serie = 0; serie < multidimwindow.length; serie++){
            XYSeries series1 = new XYSeries(labels[serie]);
            for (int frame = 0; frame < multidimwindow[0].length; frame++){
                series1.add(frame, multidimwindow[serie][frame]);
            }
            result.addSeries(series1);
        }
        return result;
    }
    private static XYSeriesCollection createJFreeData(List<Double> values){
        XYSeriesCollection result = new XYSeriesCollection();
        XYSeries series = new XYSeries("Serie");
        int frameIndex = 0;
        for (Double value: values){
            series.add(frameIndex++, value);
        }
        result.addSeries(series);
        return result;
    }
    
    private static XYSeriesCollection createJFreeData(List<Double> scores, List<Double> values){
        XYSeriesCollection result = new XYSeriesCollection();
        
        XYSeries series = new XYSeries("Values");
        int frameIndex = 0;
        double valuesmax = 0;
        for (Double value: values){
            series.add(frameIndex++, value);
            if (value > valuesmax) {
                valuesmax = value;
            }
        }
        result.addSeries(series);
        
        double scoresMax = 0;
        for (Double value: scores){
            if (value > scoresMax) {
                scoresMax = value;
            }
        }
        
        double scale = valuesmax/scoresMax;
        
        series = new XYSeries("Score");
        frameIndex = 0;
        for (Double score: scores){
            series.add(frameIndex++, Math.abs(score*scale));
        }
        result.addSeries(series);
        
        return result;
    }
    
    /**
     * Creates a chart based on the first dataset, with a fitted linear regression line.
     *
     * @return the chart panel.
     */
    private ChartPanel createChartPanel(XYDataset dataSet) {

            // create plot...
            NumberAxis xAxis = new NumberAxis("X");
            xAxis.setAutoRangeIncludesZero(false);
            NumberAxis yAxis = new NumberAxis("Y");
            yAxis.setAutoRangeIncludesZero(false);

            XYSplineRenderer renderer1 = new XYSplineRenderer();
            XYPlot plot = new XYPlot(dataSet, xAxis, yAxis, renderer1);
            plot.setBackgroundPaint(Color.lightGray);
            plot.setDomainGridlinePaint(Color.white);
            plot.setRangeGridlinePaint(Color.white);
            plot.setAxisOffset(new RectangleInsets(4, 4, 4, 4));

            // create and return the chart panel...
            JFreeChart chart = new JFreeChart("",
                    JFreeChart.DEFAULT_TITLE_FONT, plot, true);
           
            ChartUtilities.applyCurrentTheme(chart);
            ChartPanel chartPanel = new ChartPanel(chart);
            return chartPanel;
    }
    
    /**
     * Creates a new instance of the demo application.
     *
     * @param title  the frame title.
     */
    /*private PlotUtils(String title, double[][] multidimwindow) {
        super(title);
        JPanel content = createChartPanel(createJFreeData(multidimwindow));
        getContentPane().add(content);
    }*/
    private PlotUtils(String title, XYSeriesCollection values) {
        super(title);
        JPanel content = createChartPanel(values);
        getContentPane().add(content);
    }
    public static void createChart(String title, double[][] multidimwindow, String[] labels) {
        PlotUtils appFrame = new PlotUtils(title, createJFreeData(multidimwindow, labels));
        appFrame.pack();
        RefineryUtilities.centerFrameOnScreen(appFrame);
        appFrame.setVisible(true);
    }
    public static void createChart(String title, List<Double> scores, List<Double> values) {
        PlotUtils appFrame = new PlotUtils(
                title, createJFreeData(scores, values));
        appFrame.pack();
        RefineryUtilities.centerFrameOnScreen(appFrame);
        appFrame.setVisible(true);
    }
    
    
    /*public void plot(Formatter formatter) {
        boolean functionsFound;
        int level = 0;
        // create the dataset...
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        int offset;
        double d;

        do {
            functionsFound = false;
            offset = startTime;
            for (FunctionInstance x : functionInstances) {
                if (x.getOvlLevel() == level) {
                    functionsFound = true;
                    dataset.addValue(d = (double)(x.getFuncExecTimeStart() - x.getPreOverhead() - offset), "- " + Integer.toString(offset), "Level: " + Integer.toString(level));
                    dataset.addValue(d = (double)x.getPreOverhead(), "pre overhead " + Integer.toString(offset), "Level: " + Integer.toString(level));
                    dataset.addValue(d = (double)x.getFuncExecTime(), "func exec " + Integer.toString(offset), "Level: " + Integer.toString(level));
                    dataset.addValue(d = (double)x.getPostOverhead(), "post overhead " + Integer.toString(offset), "Level: " + Integer.toString(level));
                    offset = x.getFuncExecTimeStart() + x.getFuncExecTime() + x.getPostOverhead();
                }
            }
            level++;
        } while (functionsFound);

         // create the chart...
        JFreeChart chart = ChartFactory.createStackedBarChart(
            "Function hierarchy",
            "Level",                    // domain axis label
            "Cycles",                   // range axis label
            dataset,                    // data
            PlotOrientation.HORIZONTAL, // orientation
            false,                      // include legend
            true,                       // tooltips?
            false                       // URLs?
        );

        // set the background color for the chart...
        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setMargin(2.0, 0.0, 0.0, 0.0);

        // get a reference to the plot for further customisation...
        CategoryPlot plot = (CategoryPlot) chart.getPlot();

        LegendItemCollection items = new LegendItemCollection();
        items.add(new LegendItem("Function execution", null, null, null,
                new Rectangle2D.Double(-6.0, -3.0, 12.0, 6.0), Color.green));
        items.add(new LegendItem("Overhead", null, null, null,
                new Rectangle2D.Double(-6.0, -3.0, 12.0, 6.0), Color.red));
        plot.setFixedLegendItems(items);
        plot.setInsets(new RectangleInsets(5, 5, 5, 20));
        //LegendTitle legend = new LegendTitle(plot);
        //legend.setPosition(RectangleEdge.BOTTOM);
        //chart.addSubtitle(legend);

        plot.setBackgroundPaint(Color.white);
        plot.setDomainGridlinePaint(Color.black);
        plot.setDomainGridlinesVisible(true);
        //plot.setRangeGridlinePaint(Color.white);

        // set the range axis to display integers only...
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        rangeAxis.setUpperMargin(0.0);

        // disable bar outlines...
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setDrawBarOutline(false);

        // set up gradient paints for series...
        Paint gp0 = new Color(0, 0, 0, 0);
        GradientPaint gp1 = new GradientPaint(0.0f, 0.0f, Color.red,
                0.0f, 0.0f, new Color(64, 0, 0));
        GradientPaint gp2 = new GradientPaint(0.0f, 0.0f, Color.green,
                0.0f, 0.0f, new Color(0, 64, 0));
        GradientPaint gp3 = new GradientPaint(0.0f, 0.0f, Color.red,
                0.0f, 0.0f, new Color(64, 0, 0));
        for (int y = 0; y < 10; y++) {
            renderer.setSeriesPaint(y*4 + 0, gp0);
            renderer.setSeriesPaint(y*4 + 1, gp1);
            renderer.setSeriesPaint(y*4 + 2, gp2);
            renderer.setSeriesPaint(y*4 + 3, gp3);
        }

        ChartRenderingInfo info = new ChartRenderingInfo(new StandardEntityCollection());
        BufferedImage img = chart.createBufferedImage(1000, 500, info);

        try {
            OutputStream ost =
                    new BufferedOutputStream(new FileOutputStream(new File(Global.tempFileDirectory + "/" + thrInstName + ".png"), false));
            ChartUtilities.writeBufferedImageAsPNG(ost, img);
        }
        catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            return;
        }

    }*/
}
