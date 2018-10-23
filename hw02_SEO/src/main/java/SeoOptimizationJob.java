import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SeoOptimizationJob extends Configured implements Tool {
    public static class SeoOptimizationPartitioner extends Partitioner<HostQueryKey, LongWritable> {
        @Override
        public int getPartition(HostQueryKey key, LongWritable val, int num_partitions) {
            if (key.getHostName().length() == 0) {
                return 0;
            }

            float first_letter_code = key.getHostName().charAt(0);
            if (first_letter_code < 'a') {
                return 0;
            }
            if (first_letter_code > 'z') {
                return num_partitions-1;
            }

            return (int) ((first_letter_code - 'a') / ('z'-'a'+1) * num_partitions); //+1: to prevent returning more then num_partitions-1
        }
    }

    public static class SeoOptimizationMapper extends Mapper<LongWritable, Text, HostQueryKey, LongWritable> {
        private static final Pattern HostPattern = Pattern.compile("^(.*?\\:\\/+)(.*?)(\\:|\\/)");
        private final Log LOG = LogFactory.getLog(SeoOptimizationMapper.class);
        private final LongWritable One = new LongWritable(1);

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            if (parts.length != 2) {
                LOG.warn("Line with offset "+key+" has incorrect format: '"+value.toString()+"'!");
                return;
            }

            String host = "";
            try {
                URL url = new URL(parts[1]);
                host = url.getHost();
            } catch (MalformedURLException err) {
                LOG.warn("Cannot find host pattern! Line: "+value.toString()+"; host string: '"+parts[1]+"'; offset = "+key);
                return;
            }

            context.write(new HostQueryKey(host, parts[0]), One);
        }
    }

    public static class SeoOptimizationCombiner extends Reducer<HostQueryKey, LongWritable, HostQueryKey, LongWritable> {
       @Override
       protected void reduce(HostQueryKey key, Iterable<LongWritable> counts, Context context) throws IOException, InterruptedException {
           long total_count = 0;
           for (LongWritable c: counts) {
               total_count += c.get();
           }
           context.write(key, new LongWritable(total_count));
       }
    }

    public static class SeoOptimizationReducer extends Reducer<HostQueryKey, LongWritable, Text, LongWritable> {
        @Override
        protected void reduce(HostQueryKey key, Iterable<LongWritable> counts, Context context) throws IOException, InterruptedException {
            String max_query_text = "";
            long max_query_count = 0;

            long current_query_count = 0;
            String current_query = "";

            for (LongWritable c: counts) {
                String query = key.getQuery();
                if ((!query.equals(current_query)) && (current_query_count != 0)) {
                    if (current_query_count > max_query_count) {
                        max_query_text = current_query;
                        max_query_count = current_query_count;
                    }

                    current_query_count = c.get();
                    current_query = query;
                }
                else {
                    if (current_query_count == 0) {
                        current_query = query;
                    }
                    current_query_count += c.get();
                }
            }

            if (current_query_count > max_query_count) {
                max_query_text = current_query;
                max_query_count = current_query_count;
            }
            if (max_query_count >= getMinClicks(context.getConfiguration())) {
                context.write(new Text(key.getHostName()+"\t"+max_query_text), new LongWritable(max_query_count));
            }
        }


        public static final String MINCLICKS_PARAMETER = "seo.minclicks";
        public static long getMinClicks(Configuration conf) {
            return conf.getLong(MINCLICKS_PARAMETER, 100);
        }
    }

    private static final int NUM_REDICERS = 15;
    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(SeoOptimizationJob.class);
        job.setJobName(SeoOptimizationJob.class.getCanonicalName());

        job.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.setPartitionerClass(SeoOptimizationPartitioner.class);
        job.setMapperClass(SeoOptimizationMapper.class);
        job.setReducerClass(SeoOptimizationReducer.class);
        job.setCombinerClass(SeoOptimizationCombiner.class);

        job.setNumReduceTasks(NUM_REDICERS);

        job.setSortComparatorClass(HostQueryKey.Compator.class);
        job.setGroupingComparatorClass(HostQueryKey.GroupComparator.class);

        job.setMapOutputKeyClass(HostQueryKey.class);
        job.setMapOutputValueClass(LongWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new SeoOptimizationJob(), args);
        System.exit(ret);
    }

}
