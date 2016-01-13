package org.apache.mahout.classifier.sequencelearning.hmm.hadoop;

import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmEvaluator;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * This class implements a program that models a HMM to identify CpG islands
 * in a given DNA sequence. The HMM is trained using Baum Welch in mapreduce.
 * Baum Welch is unsupervised training which takes a DNA sequence as input
 * for training the HMM.
 */

public class CpGIslandFinder {
    private static final Logger log = LoggerFactory.getLogger(CpGIslandFinder.class);

    /**
     * Model trained in the example.
     */
    private static HmmModel trainedModel;
    
    /**
     * Configuration object
     */
    private static Configuration conf = new Configuration();

    /**
     * Filesystem of the cluster
     */
    private static FileSystem fs;

    /**
     * Number of hidden states
     */
    private static int numHiddenStates = 8;
    private static int numEmissionSymbols = 4;

    /**
     * Path to the initial Hmm Model
     */
    private static Path modelIn = new Path("tmp/modelIn");

    /**
     * Path to the input file
     */
    private static Path input = new Path("tmp/input");

    /**
     * Path to the output directory
     */
    private static Path output = new Path("tmp/output");

    /**
     * Path to the output model directory
     */
    private static Path modelOut = new Path("tmp/modelOut");

    /**
     * Path to the (hidden state => id) map
     */
    private static Path hiddenStatesMapPath = new Path("tmp/maps/hiddenStatesMap");

    /**
     * Path to the (emitted state => id) map
     */
    private static Path emittedStatesMapPath = new Path("tmp/maps/emittedStatesMap");

    private static void setUpBaumWelch() {
	conf.set(BaumWelchConfigKeys.SCALING_OPTION_KEY, "rescaling");
	conf.set(BaumWelchConfigKeys.NUMBER_OF_HIDDEN_STATES_KEY, "8");
	conf.set(BaumWelchConfigKeys.NUMBER_OF_EMITTED_STATES_KEY, "4");
	conf.set(BaumWelchConfigKeys.MODEL_PATH_KEY, "tmp/modelIn");
	conf.set(BaumWelchConfigKeys.MODEL_CONVERGENCE_KEY, ".005");
	conf.set(BaumWelchConfigKeys.HIDDEN_STATES_MAP_PATH, "tmp/maps/hiddenStatesMap");
	conf.set(BaumWelchConfigKeys.EMITTED_STATES_MAP_PATH, "tmp/maps/emittedStatesMap");

    }

    private static void trainModel(String trainingPath, String trainedHmmFile, int numIter, String convergence) throws Exception {
	List<Integer> observedSequence = new LinkedList<Integer>();
	BufferedReader reader = new BufferedReader(new FileReader(trainingPath));
	int chr;
	int val;
	int count = 0;
	int chunk = 1;
	fs  = FileSystem.get(conf);
	SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, input, LongWritable.class, VectorWritable.class);
	try {
		while ( (chr = reader.read()) != -1)
	    {
			if ((chr == 'A') || (chr == 'a'))
				val = 0;
			else if ((chr == 'C') || (chr == 'c'))
				val = 1;
			else if ((chr == 'G') || (chr == 'g'))
				val = 2;
			else if ((chr == 'T') || (chr == 't'))
				val = 3;
			else
				val = -1;

			if (val != -1) {
				observedSequence.add(val);
				count++;
			}

			if ((count != 0) && (count % 0x10000 == 0)) {
			    Vector inputVector = new DenseVector(0x10000);
			    int cnt = 0;
			    for (int val1 : observedSequence) {
					inputVector.set(cnt++, (double) val1);
			    }
			    observedSequence.clear();
			    VectorWritable inputVectorWritable = new VectorWritable(inputVector);


				writer.append(new LongWritable(chunk++), inputVectorWritable);
			}
	    }
	} finally {
		Closeables.closeQuietly(writer);
	}

    log.info("Size of input file:"+count);

    

    count = 0;
    
    // BaumWelchUtils.buildRandomModel(numHiddenStates, numEmissionSymbols, modelIn, conf);

    double[] initialP = {0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2};

    double[][] transitionP = {{0.170, 0.274, 0.426, 0.120, 0.0025, 0.0025, 0.0025, 0.0025},
				  {0.170, 0.358, 0.274, 0.188, 0.0025, 0.0025, 0.0025, 0.0025},
				  {0.161, 0.329, 0.375, 0.125, 0.0025, 0.0025, 0.0025, 0.0025},
				  {0.079, 0.345, 0.384, 0.182, 0.0025, 0.0025, 0.0025, 0.0025},
				  {0.0025, 0.0025, 0.0025, 0.0025, 0.300, 0.205, 0.275, 0.210},
				  {0.0025, 0.0025, 0.0025, 0.0025, 0.393, 0.137, 0.088, 0.372 },
				  {0.0025, 0.0025, 0.0025, 0.0025, 0.248, 0.246, 0.288, 0.208},
				  {0.0025, 0.0025, 0.0025, 0.0025, 0.177, 0.239, 0.282, 0.292}};

    double[][] emissionP =   {{1.0, 0.0, 0.0, 0.0},
			      {0.0, 1.0, 0.0, 0.0},
			      {0.0, 0.0, 1.0, 0.0},
			      {0.0, 0.0, 0.0, 1.0},
			      {1.0, 0.0, 0.0, 0.0},
			      {0.0, 1.0, 0.0, 0.0},
			      {0.0, 0.0, 1.0, 0.0},
			      {0.0, 0.0, 0.0, 1.0}};


    BaumWelchUtils.buildHmmModelFromDistributions(initialP, transitionP, emissionP,
						      modelIn, conf);

    MapWritable hiddenMap = new MapWritable();
    MapWritable emittedMap = new MapWritable();

    hiddenMap.put(new Text("A+"), new IntWritable(0));
    hiddenMap.put(new Text("C+"), new IntWritable(1));
    hiddenMap.put(new Text("G+"), new IntWritable(2));
    hiddenMap.put(new Text("T+"), new IntWritable(3));
    hiddenMap.put(new Text("A-"), new IntWritable(4));
    hiddenMap.put(new Text("C-"), new IntWritable(5));
    hiddenMap.put(new Text("G-"), new IntWritable(6));
    hiddenMap.put(new Text("T-"), new IntWritable(7));

    emittedMap.put(new Text("a"), new IntWritable(0));
    emittedMap.put(new Text("c"), new IntWritable(1));
    emittedMap.put(new Text("g"), new IntWritable(2));
    emittedMap.put(new Text("t"), new IntWritable(3));


    MapWritableCache.save(new IntWritable(1), hiddenMap, hiddenStatesMapPath, conf);
    MapWritableCache.save(new IntWritable(1), emittedMap, emittedStatesMapPath, conf);

    modelOut = BaumWelchDriver.runBaumWelchMR(conf, input, modelIn, output, hiddenStatesMapPath,
					      emittedStatesMapPath, numHiddenStates, numEmissionSymbols, convergence, "rescaling", numIter);

    trainedModel =  BaumWelchUtils.createHmmModel(8, 4, modelOut, conf);
    Vector initialProbabilities = trainedModel.getInitialProbabilities();
    Matrix emissionMatrix = trainedModel.getEmissionMatrix();
    Matrix transitionMatrix = trainedModel.getTransitionMatrix();
    BufferedWriter writer1 = new BufferedWriter(new FileWriter(trainedHmmFile));
    for (int i = 0; i < trainedModel.getNrOfHiddenStates(); ++i) {
	writer1.write(Double.toString(initialProbabilities.get(i)));
	writer1.newLine();

	for (int j = 0; j < trainedModel.getNrOfHiddenStates(); ++j) {
	    writer1.write(Double.toString(transitionMatrix.get(i, j)));
	    writer1.write(" ");
	}
	writer1.newLine();
	
	for (int j = 0; j < trainedModel.getNrOfOutputStates(); ++j) {
	    writer1.write(Double.toString(emissionMatrix.get(i, j)));
	    writer1.write(" ");
	}
	writer1.newLine();
    }
    writer1.close();
    }

    private static void testModel(String testFilePath, String stateSeqFile) throws Exception {
	log.info("testing");
	List<Integer> observedSequence = new LinkedList<Integer>();
	int[] testSequence = new int[0x100000];
	
	BufferedReader reader = new BufferedReader(new FileReader(testFilePath));
	BufferedWriter writer1 = new BufferedWriter(new FileWriter(stateSeqFile));
	int chr;
	int val;
	int count = 0;
	int chunk = 0;
	while ( (chr = reader.read()) != -1)
	    {
			if ((chr == 'A') || (chr == 'a'))
				val = 0;
			else if ((chr == 'C') || (chr == 'c'))
				val = 1;
			else if ((chr == 'G') || (chr == 'g'))
				val = 2;
			else if ((chr == 'T') || (chr == 't'))
				val = 3;
			else
				val = -1;

		if (val != -1) {
		    observedSequence.add(val);
		    count++;
		}

		if ((count != 0) && (count % 0x100000 == 0)) {
			for (int i = 0; i < 0x100000; i++)
			    testSequence[i] = observedSequence.get(i);
			observedSequence.clear();
			int[] stateSeq = HmmEvaluator.decode(trainedModel, testSequence, true);

			int beg=0;
			boolean inIsland = false;
			int cCount = 0;
			int gCount = 0;
			int cgCount = 0;
			int islandLen = 0;
			boolean atC = false;
			for (int i = 0; i < 0x100000; ++i) {
			    val = stateSeq[i];
			    if (inIsland)
				{
				    if ( (val == 4) || (val == 5) || (val == 6) || (val == 7))
					{
					    inIsland = false;

					    int end = i - 1;
					    double ccnt = cCount;
					    double gcnt = gCount;
					    double cgcontent = (ccnt + gcnt)/islandLen;
					    double oeratio = 0.0;
					    if ((cCount != 0) && (gCount != 0))
						oeratio = (cgCount * islandLen)/(ccnt * gcnt);

					    if (/*(islandLen > 200) && */(cgcontent > 0.5) && (oeratio > 0.6))
						{
						    writer1.write(String.format("%d %d %d %f %f\n", beg + chunk*0x100000 +1, end + chunk*0x100000 +1,
										islandLen, cgcontent, oeratio));
						}
					    else
						{
						    /*writer1.write(String.format("False Positive: %d %d %d %d %f %f\n",
						      islandLen, cCount, gCount, cgCount, cgcontent, oeratio));*/
						}
					}
				    else
					{
					    islandLen++;
					    if (val == 2)
						{
						    gCount++;
						    if (atC)
							{
							    cgCount++;
							}
						}

					    if (val == 1)
						{
						    cCount++;
						    atC = true;
						}
					    else
						atC = false;
					}
				}
			    else
				{
				    if ( (val == 0) || (val == 1) || (val == 2) || (val == 3))
					{
					    inIsland = true;
					    islandLen = 1;
					    cgCount = 0;
					    beg = i;
					    if (val == 1)
						{
						    cCount = 1;
						    atC = true;
						}
					    else
						cCount = 0;

					    if (val == 2)
						gCount = 1;
					    else
						gCount = 0;
					}
				}
			}
			chunk++;
		}
	    }
	writer1.close();
    }

    public static void main(String[] args) throws Exception {
	String trainingFile = args[0];
	String testFile = args[1];
	String stateSeqFile = args[2];
	String trainedHmmFile = args[3];
	String convergence = args[4];
	int numIter = Integer.parseInt(args[5]);

	setUpBaumWelch();
	trainModel(trainingFile, trainedHmmFile, numIter, convergence);
	testModel(testFile, stateSeqFile);
    }
}
