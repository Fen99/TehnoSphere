import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

public class HITSJob extends Configured implements Tool {
    // <id> <auth> <hub> <children>
    public static class HITSMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private final Log LOG = LogFactory.getLog(HITSMapper.class);
        private HashMap<Integer, Long> auth_values_ = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException {
            Integer current_iteration = Integer.parseInt(context.getConfiguration().get("iteration"));
            String input_path = ((FileSplit) context.getInputSplit()).getPath().toString();
            if (current_iteration != 0) {
                input_path = input_path.replace(current_iteration+"/part", (current_iteration-1)+"/part"); // Получаем путь до предпоследней итерации
            }
            Path input = new Path(input_path);
            FileSystem fs = input.getFileSystem(context.getConfiguration());
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(input)));

            String line = reader.readLine();
            while (line != null) {
                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split("\t");
                Integer id = Integer.parseInt(parts[0]);
                Long auth = 0L;
                if (current_iteration != 0) {
                    auth = Long.parseLong(parts[1]);
                } else {
                    auth = 1L;
                }
                auth_values_.put(id, auth);
                line = reader.readLine();
            }
            reader.close();
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            IntWritable doc_id = new IntWritable(Integer.parseInt(parts[0]));
            String next_nodes_str = "";
            Long our_auth = 0L;
            Long our_hub = 0L;

            boolean is_first = false;
            if (parts.length == 2) { // Первая итерация HITS
                next_nodes_str = parts[1];
                our_hub = 1L;
                our_auth = 1L;
                is_first = true;
            } else {
                our_auth = Long.parseLong(parts[1]);
                our_hub = Long.parseLong(parts[2]);
                next_nodes_str = parts[3];
            }

            ArrayList<Integer> next_nodes = PageRankJob.PageRankMapper.GetNextNodes(next_nodes_str);
            Long next_hub = 0l;
            if (is_first) {
                next_hub = 1L * next_nodes.size();
            } else {
                for (Integer node : next_nodes) {
                    next_hub += auth_values_.get(node);
                }
            }

            context.write(doc_id, new Text(our_auth+"\t"+our_hub+"\t"+next_nodes_str)); //Посылаем старые значения
            context.write(doc_id, new Text("H"+next_hub)); // Посылаем новую "хабность"
            for (Integer next_node_id : next_nodes) {
                context.write(new IntWritable(next_node_id), new Text(our_hub.toString())); // Рассылаем всем хабам нашу старую хабность
            }
        }
    }

    public static class HITSReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String next_nodes = "";
            Long prev_auth = 0L;
            Long new_auth = 0L;
            Long prev_hub = 0L;
            Long new_hub = 0L;

            for (Text val: values) {
                String val_str = val.toString();
                if (val_str.lastIndexOf('\t') != -1) { // Старая auth + старый hub + список наследников
                    String[] splits = val_str.split("\t");
                    next_nodes = splits[2];
                    prev_auth = Long.parseLong(splits[0]);
                    prev_hub = Long.parseLong(splits[1]);
                } else if (val_str.startsWith("H")) { // Посылка новой hub-ности
                    new_hub = Long.parseLong(val_str.substring(1));
                } else { // Посылается значение для увеличения auth
                    new_auth += Long.parseLong(val_str);
                }
            }

            if (next_nodes.isEmpty()) {
                next_nodes = PageRankJob.PageRankMapper.OrphanPrefix;
            }
            context.write(key, new Text(new_auth+"\t"+new_hub+"\t"+next_nodes));
        }
    }

    private static final int NUM_REDICERS = 1;
    private static final int MAX_ITERATIONS = 3;
    private Job getJobConf(String input, String output, Integer iteration) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(HITSJob.class);
        job.setJobName(HITSJob.class.getCanonicalName());
        job.getConfiguration().set("iteration", iteration.toString());

        job.setInputFormatClass(TextInputFormat.class);
        if (iteration == 0) {
            FileInputFormat.addInputPath(job, new Path(input));
        } else {
            FileInputFormat.addInputPath(job, new Path(output+(iteration-1)+"/part-*"));
        }
        FileOutputFormat.setOutputPath(job, new Path(output+iteration));

        job.setMapperClass(HITSMapper.class);
        job.setReducerClass(HITSReducer.class);

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
        }
        return 0;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new HITSJob(), args);
        System.exit(ret);
    }
}