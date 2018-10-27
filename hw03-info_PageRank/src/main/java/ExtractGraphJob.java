import org.apache.commons.codec.binary.Base64;
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
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.yarn.webapp.view.HtmlPage;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

public class ExtractGraphJob extends Configured implements Tool {
    public static class ExtractGraphMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final Pattern HrefPattern = Pattern.compile("(<a.*?href=\")(.*?)(\")");

        private final Log LOG = LogFactory.getLog(ExtractGraphMapper.class);
        private static final IntWritable One = new IntWritable(1);

        public static final String PagesURLsFilePath = "/data/infopoisk/hits_pagerank/urls.txt"; // Путь до urls.txt
        public static final String URLsIndexPath = "/user/f.petryajkin/homeworks/hw03-info/links_list/part-r-00000"; // Пустое, если строим файл с id ссылок, иначе - путь до этого файла

        private HashMap<Integer, String> documents_urls_ = new HashMap<>(); // Соответствие id --> URL (для чтения документов)
        private HashMap<String, Integer> urls_dict_ = new HashMap<>(); // Соответствие URL --> id (для построения графа)

        private String NormalizeURL(String base_url, String url) {
            url = url.replace("https:", "http:");
            url = url.replace("www.", "");
            url = url.replace(" ", "");

            URI base;
            try {
                base = new URI(base_url);
            } catch (URISyntaxException exc) {
                LOG.warn("Wrong base address format: "+base_url);
                return null;
            }
            String result = url;
            if (result.startsWith("//")) {
                result = "http:"+result;
            } else if (result.startsWith("/")) {
                result = "http://"+base.getHost()+result;
            } else if (!result.startsWith("http:")) {
                result = base_url+"/"+result;
            }

            if (result.charAt(result.length()-1) == '/') {
                result = result.substring(0, result.length()-1);
            }

            URI curr_url;
            try {
                curr_url = new URI(result);
            } catch (URISyntaxException exc) {
                LOG.warn("Wrong url format: "+result);
                return null;
            }

            curr_url = curr_url.normalize();
            if (curr_url == null) {
                return null;
            } else if (curr_url.getHost() == null) {
                return null;
            } else if (curr_url.getHost().lastIndexOf("lenta.ru") == -1) { // Сайт не с lenta.ru
                return null;
            }

            return curr_url.toString().toLowerCase();
        }

        // Читает из входного файла все строки вида id\tURL в словарь, URL нормализуются
        // reverse = false => строится id --> URL, иначе URL --> id
        private void ReadAllUrls(Context context, String filename,
                                 Object destination, boolean reverse)
                throws IOException {
            Path urls = new Path(filename);
            FileSystem fs = urls.getFileSystem(context.getConfiguration());
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(urls)));
            String line = reader.readLine();
            while (line != null && line != "") {
                String[] parts = line.split("\t");
                Integer id = Integer.parseInt(parts[0]);
                String url_norm = NormalizeURL("", parts[1]);
                if (url_norm == null) {
                    continue;
                }

                if (!reverse) {
                    ((HashMap<Integer, String>) destination).put(id, url_norm);
                } else {
                    ((HashMap<String, Integer>) destination).put(url_norm, id);
                }

                line = reader.readLine();
            }

            reader.close();
        }

        @Override
        protected void setup(Context context) throws IOException {
            ReadAllUrls(context, PagesURLsFilePath, documents_urls_, false);
            if (URLsIndexPath != "") {
                ReadAllUrls(context, URLsIndexPath, urls_dict_, true);
            }
        }

        // Возвращает все URL из блоков href в нормализованном виде
        private HashSet<String> GetLinks(String base_url, String document_text) {
            HashSet<String> links = new HashSet<>();
            Matcher m = HrefPattern.matcher(document_text);
            while (m.find()) {
                if (m.group().lastIndexOf("mailto:") != -1) {
                    continue;
                }

                String normalized = NormalizeURL(base_url, m.group(2)); // Проверяет принадлежность lenta.ru, нормализует URL
                if (normalized != null) {
                    links.add(normalized);
                }
            }
            return links;
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            final int buffer_size = 2048;

            String[] parts = value.toString().split("\t");
            Integer document_id = Integer.parseInt(parts[0]);

            Inflater document_parser = new Inflater();
            document_parser.setInput(Base64.decodeBase64(parts[1].getBytes()));
            document_parser.finished();
            byte[] buffer = new byte[buffer_size];
            String document = "";
            while (true) {
                try {
                    int decompressed_len = document_parser.inflate(buffer);
                    if (decompressed_len > 0) {
                        document += new String(buffer, 0, decompressed_len);
                    } else {
                        break;
                    }
                } catch (DataFormatException exc) {
                    LOG.warn("Incorrect document format! ID = "+document_id+"; offset = "+key);
                    break;
                }
            }

            HashSet<String> links = GetLinks(documents_urls_.get(document_id), document);
            if (URLsIndexPath == "") { // Строим индекс ссылок
                context.write(new Text(documents_urls_.get(document_id)), One);
                for (String link: links) {
                    context.write(new Text(link), One);
                }
            }
            else { // Строим граф
                Integer document_link_id = urls_dict_.get(documents_urls_.get(document_id));
                for (String link: links) {
                    Integer current_link_id = urls_dict_.get(link);
                    if (current_link_id == null) {
                        LOG.warn("Unknown link! "+link);
                        continue;
                    }
                    context.write(new Text(document_link_id.toString()), new IntWritable(current_link_id));
                }
            }
        }
    }

    public static class ExtractGraphReducer extends Reducer<Text, IntWritable, Text, Text> {
        private Integer url_id_ = 0;

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            if (ExtractGraphMapper.URLsIndexPath == "") { // Строим индекс ссылок
                context.write(new Text(url_id_.toString()), key);
                url_id_++;
            } else {
                String result = "";
                for (IntWritable val: values) {
                    result += val.get()+" ";
                }
                context.write(key, new Text(result.substring(0, result.length()-1)));
            }
        }
    }

    private static final int NUM_REDICERS = 1;
    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(ExtractGraphJob.class);
        if (ExtractGraphMapper.URLsIndexPath != "") {
            job.setJobName(ExtractGraphJob.class.getCanonicalName());
        } else {
            job.setJobName("LinksExtractor");
        }

        job.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        job.setMapperClass(ExtractGraphMapper.class);
        job.setReducerClass(ExtractGraphReducer.class);

        job.setNumReduceTasks(NUM_REDICERS);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new ExtractGraphJob(), args);
        System.exit(ret);
    }

}