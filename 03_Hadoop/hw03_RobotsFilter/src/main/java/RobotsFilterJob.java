import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.client.Delete;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableReducer;
import org.apache.hadoop.hbase.mapreduce.TableSplit;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import javax.xml.bind.DatatypeConverter;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

public class RobotsFilterJob extends Configured implements Tool {
    private static class HostFileExtractor {
        private String host_ = "";
        private String file_ = "";

        public HostFileExtractor(String url) {
            int protocol_pos = url.indexOf("://");
            if (protocol_pos != -1) {
                url = url.substring(protocol_pos+3);
            }

            String[] parts = url.split("/", 2);
            if (parts.length == 1) {
                host_ = url;
            } else {
                host_ = parts[0];
                file_ = parts[1];
                if (!file_.startsWith("/")) {
                    file_ = "/"+file_;
                }
            }
        }

        public String GetHost() {
            return host_;
        }

        public String GetFile() {
            return file_;
        }
    }

    private static final String WebsitesTableNameParameter = "websites_name";
    private static final String WebpagesTableNameParameter = "webpages_name";
    private static final String DisabledLabel = "#D#";
    private static final String EnabledLabel = "#E#";

    // Table -> <host, U+url> (из таблицы webpages) или <host, R+robots> (из таблицы websites)
    public static class RobotsFilterMapper extends TableMapper<LabelHostKey, Text> {
        @Override
        protected void map(ImmutableBytesWritable rowKey, Result columns, Context context) throws IOException, InterruptedException {
            TableSplit current_split = (TableSplit) context.getInputSplit();
            String table_name = new String(current_split.getTableName());

            String websites_table_name = context.getConfiguration().get(WebsitesTableNameParameter);
            String webpages_table_name = context.getConfiguration().get(WebpagesTableNameParameter);

            if (table_name.equals(websites_table_name)) {
                String site_name = new String(columns.getValue(Bytes.toBytes("info"), Bytes.toBytes("site")));

                byte[] robots_bytes = columns.getValue(Bytes.toBytes("info"), Bytes.toBytes("robots"));
                String robots = "";
                if (robots_bytes != null) {
                    robots = new String(columns.getValue(Bytes.toBytes("info"), Bytes.toBytes("robots")));
                }
                 context.write(new LabelHostKey(true, site_name), new Text(robots));
            }

            else if (table_name.equals(webpages_table_name)) {
                String url_string = new String(columns.getValue(Bytes.toBytes("docs"), Bytes.toBytes("url")));
                HostFileExtractor url = new HostFileExtractor(url_string);
                boolean is_enabled = (columns.getValue(Bytes.toBytes("docs"), Bytes.toBytes("disabled")) == null);

                if (is_enabled) {
                    url_string = EnabledLabel+url_string;
                } else {
                    url_string = DisabledLabel+url_string;
                }
                context.write(new LabelHostKey(false, url.GetHost()), new Text(url_string));
            }

            else {
                throw new InterruptedException("Unknown table: "+table_name+"; "+webpages_table_name+"; "+websites_table_name);
            }
        }
    }

    public static class RobotsFilterPartitioner extends Partitioner<LabelHostKey, Text> {
        @Override
        public int getPartition(LabelHostKey key, Text val, int num_partitions) {
            return key.GetHash() % num_partitions;
        }
    }

    // <Text, [Text]> -> <url, Y/N>
    public static class RobotsFilterReducer extends TableReducer<LabelHostKey, Text, ImmutableBytesWritable> {
        private static byte[] GetMD5Hash(String for_hash) throws InterruptedException {
            byte[] bytesOfMessage = for_hash.getBytes();

            MessageDigest md = null;
            try {
                md = MessageDigest.getInstance("MD5");
            } catch (NoSuchAlgorithmException exc) {
                throw new InterruptedException(exc.getMessage());
            }
            byte[] digest = md.digest(bytesOfMessage);
            return DatatypeConverter.printHexBinary(digest).toLowerCase().getBytes();
        }

        private static boolean IsDisabled(RobotsFilter filter, String url_string) throws InterruptedException {
            HostFileExtractor extractor = new HostFileExtractor(url_string);
            String for_check = extractor.GetFile();
            if (for_check == null) { // Cannot extract url
                return false;
            }

            boolean disabled = false;
            if (filter != null) {
                try {
                    if (!filter.IsAllowed(for_check)) {
                        disabled = true;
                    }
                } catch (RobotsFilter.BadFormatException exc) {
                    throw new InterruptedException(exc.getMessage());
                }
            }
            return disabled;
        }

        @Override
        protected void reduce(LabelHostKey website, Iterable<Text> vals, Context context) throws IOException, InterruptedException {
            RobotsFilter filter = null;
            int iteration = 0;

            for (Text value: vals) {
                String current_str = value.toString();

                if (website.IsRobots()) {
                    if (iteration != 0) {
                        throw new InterruptedException("robots.txt should be first in the list!");
                    }
                    if (filter != null) {
                        throw new InterruptedException("Too more robots.txt for one site!");
                    }

                    try {
                        filter = new RobotsFilter(current_str);
                    } catch (RobotsFilter.BadFormatException exc) {
                        throw new InterruptedException(exc.getMessage());
                    }
                    continue;
                }
                iteration++;

                boolean is_disabled_on_start = current_str.startsWith(DisabledLabel);
                current_str = current_str.substring(DisabledLabel.length());
                byte[] hash = GetMD5Hash(current_str);

                boolean disabled = IsDisabled(filter, current_str);
                if (disabled && !is_disabled_on_start) {
                    Put put = new Put(hash);
                    put.add(Bytes.toBytes("docs"), Bytes.toBytes("disabled"), Bytes.toBytes("Y"));
                    context.write(null, put);
                } else if (!disabled && is_disabled_on_start) {
                    Delete del = new Delete(hash);
                    del.deleteColumn(Bytes.toBytes("docs"), Bytes.toBytes("disabled"));
                    context.write(null, del);
                }
            }
        }
    }

    private List<Scan> GetScans(String[] input_names) {
        List<Scan> scans = new ArrayList<>();

        for (String name: input_names) {
            Scan scan = new Scan();
            scan.setAttribute("scan.attributes.table.name", Bytes.toBytes(name));
            scans.add(scan);
        }
        return scans;
    }

    private static final int CountReducers = 2;
    private Job getJobConf(String webpages_name, String websites_name) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(RobotsFilterJob.class);
        job.setJobName(RobotsFilterJob.class.getCanonicalName());

        Configuration conf = job.getConfiguration();
        conf.set(WebpagesTableNameParameter, webpages_name);
        conf.set(WebsitesTableNameParameter, websites_name);

        job.setNumReduceTasks(CountReducers);

        String[] names = {webpages_name, websites_name};
        List<Scan> scans = GetScans(names);
        TableMapReduceUtil.initTableMapperJob(scans, RobotsFilterMapper.class, LabelHostKey.class, Text.class, job);
        TableMapReduceUtil.initTableReducerJob(webpages_name, RobotsFilterReducer.class, job);

        job.setGroupingComparatorClass(LabelHostKey.GroupComparator.class);
        job.setSortComparatorClass(LabelHostKey.Compator.class);
        job.setPartitionerClass(RobotsFilterPartitioner.class);

        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new RobotsFilterJob(), args);
        System.exit(ret);
    }
}