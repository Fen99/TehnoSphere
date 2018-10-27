import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.ArrayList;

public class PageRankJob extends Configured implements Tool {
    public final static int CountVertices = 2891873;
    public final static int StartCountVertices = 564548;
    private final static double Alpha = 0.9;

    public final static long Normalizator = (long) 1e12;
    private enum COUNTERS {
        ORPHAN_VERTICES_RANK,
        AVGERAGE_DELTA_PR,
        MAXIMAL_DELTA_PR
    }

    public static class PageRankMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        public static final String OrphanPrefix = "ORPHAN";
        private final Log LOG = LogFactory.getLog(PageRankMapper.class);

        public static ArrayList<Integer> GetNextNodes(String next_nodes) {
            ArrayList<Integer> result = new ArrayList<>();
            if (next_nodes.equals(OrphanPrefix)) {
                return result;
            }

            for (String s: next_nodes.split(" ")) {
                if (s.isEmpty()) {
                    continue;
                }
                result.add(Integer.parseInt(s));
            }
            return result;
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            IntWritable doc_id = new IntWritable(Integer.parseInt(parts[0]));
            String next_nodes_str = "";
            double our_pr = 0.0;

            if (parts.length == 2) { // Первая итерация PR
                our_pr = 1.0 / StartCountVertices; // Начинаем блуждание с этого количества вершин
                next_nodes_str = parts[1];
            } else {
                our_pr = Double.parseDouble(parts[1]);
                next_nodes_str = parts[2];
            }

            ArrayList<Integer> next_nodes = GetNextNodes(next_nodes_str);
            context.write(doc_id, new Text(our_pr+"\t"+next_nodes_str));
            if (next_nodes.size() != 0) {
                Double next_node_pr = our_pr / next_nodes.size();
                for (Integer next_node_id : next_nodes) {
                    context.write(new IntWritable(next_node_id), new Text(next_node_pr.toString()));
                }
            } else { // Висячая вершина
                context.getCounter(COUNTERS.ORPHAN_VERTICES_RANK).increment((long) (Normalizator * our_pr));
            }
        }
    }

    public static class PageRankReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double rank_from_orhan = ((double) context.getCounter(COUNTERS.ORPHAN_VERTICES_RANK).getValue()) / Normalizator / CountVertices;
            double final_rank = (1.0 - Alpha)*(1.0 / CountVertices) + Alpha*rank_from_orhan; // Телепортация + "оставшийся" PR

            String next_nodes = "";
            double prev_pr = 0.0;
            for (Text val: values) {
                String val_str = val.toString();
                if (val_str.lastIndexOf('\t') != -1) { // Список наследников + предыдущий PR
                    String[] splits = val_str.split("\t");
                    next_nodes = splits[1];
                    prev_pr = Double.parseDouble(splits[0]);
                } else {
                    final_rank += Alpha * Double.parseDouble(val_str); // PR от других вершин
                }
            }

            if (next_nodes.isEmpty()) {
                next_nodes = PageRankMapper.OrphanPrefix;
            }
            context.write(key, new Text(final_rank+"\t"+next_nodes));
            if (!next_nodes.isEmpty()) {
                context.getCounter(COUNTERS.AVGERAGE_DELTA_PR).increment((long) (Normalizator * Math.abs(prev_pr - final_rank) / prev_pr));
            }
        }
    }

    private static final int NUM_REDICERS = 1;
    private static final int MAX_ITERATIONS = 15;
    private static final double MIN_AVG_PR_DELTA = 0.005;

    private Job getJobConf(String input, String output, int iteration) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(PageRankJob.class);
        job.setJobName(PageRankJob.class.getCanonicalName());

        job.setInputFormatClass(TextInputFormat.class);
        if (iteration == 0) {
            FileInputFormat.addInputPath(job, new Path(input));
        } else {
            FileInputFormat.addInputPath(job, new Path(output+(iteration-1)+"/part-*"));
        }
        FileOutputFormat.setOutputPath(job, new Path(output+iteration));

        job.setMapperClass(PageRankMapper.class);
        job.setReducerClass(PageRankReducer.class);

        job.setNumReduceTasks(NUM_REDICERS);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            Job job = getJobConf(args[0], args[1], i);
            if (!job.waitForCompletion(true)) {
                return 1;
            }
            if (((double) job.getCounters().findCounter(COUNTERS.AVGERAGE_DELTA_PR).getValue()) / StartCountVertices / Normalizator < MIN_AVG_PR_DELTA) { // Не по висячим вершинам
                break;
            }
        }
        return 0;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new PageRankJob(), args);
        System.exit(ret);
    }
}