import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DFCountJob extends Configured implements Tool {
    public static class DFCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        static final IntWritable one = new IntWritable(1);
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String text = value.toString();
            // Get words
            Matcher matcher = Pattern.compile("\\p{L}+").matcher(text);
            Set<String> unique_words = new HashSet<>();
            while (matcher.find()) {
                String current_word = matcher.group().toLowerCase();
                if (!unique_words.contains(current_word)) {
                    unique_words.add(current_word);
                    context.write(new Text(current_word), one);
                }
            }
        }
    }

    public static class DFCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(Text word, Iterable<IntWritable> nums, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for(IntWritable i: nums) {
                sum += i.get();
            }

            // produce pairs of "word" <-> amount
            context.write(word, new IntWritable(sum));
        }
    }

    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(DFCountJob.class);
        job.setJobName(DFCountJob.class.getCanonicalName());

        job.setInputFormatClass(DeflateInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.setMapperClass(DFCountMapper.class);
        job.setCombinerClass(DFCountReducer.class);
        job.setReducerClass(DFCountReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new DFCountJob(), args);
        System.exit(ret);
    }
}
