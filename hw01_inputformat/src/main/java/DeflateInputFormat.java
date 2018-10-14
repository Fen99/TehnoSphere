import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

public class DeflateInputFormat extends FileInputFormat<LongWritable, Text> {
    public static class DeflateFileSplit extends FileSplit {
        private List<Integer> documents_sizes_;

        public DeflateFileSplit() {
            super();
        }

        public DeflateFileSplit(Path file, long start, long split_len, List<Integer> document_sizes) {
            super(file, start, split_len, new String[]{});
            documents_sizes_ = new ArrayList<>(document_sizes);
        }

        @Override
        public void write(DataOutput out) throws IOException {
            super.write(out);

            out.writeInt(documents_sizes_.size());
            for (Integer page_len : documents_sizes_) {
                out.writeInt(page_len);
            }
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            super.readFields(in);

            documents_sizes_ = new ArrayList<>();
            for (int count_sizes = in.readInt(); count_sizes > 0; count_sizes--) {
                documents_sizes_.add(in.readInt());
            }
        }

        public List<Integer> getDocumentSizes() {
            return documents_sizes_;
        }
    }

    public static class DeflateRecordReader extends RecordReader<LongWritable, Text> {
        private FSDataInputStream input_stream_;
        private Integer current_offset_ = 0;
        private List<Integer> documents_sizes_;
        private int current_document_ = 0;
        private Text current_document_text_ = new Text();

        private final Log LOG = LogFactory.getLog(DeflateRecordReader.class);
        private final int MAX_BUFFER_SIZE = 2048;

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException {
            Configuration conf = context.getConfiguration();
            DeflateFileSplit deflate_split = (DeflateFileSplit) split;
            documents_sizes_ = deflate_split.getDocumentSizes();

            Path path = deflate_split.getPath();
            FileSystem fs = path.getFileSystem(conf);
            input_stream_ = fs.open(path);
            current_offset_ = (int) deflate_split.getStart();
            input_stream_.seek(current_offset_);
        }

        @Override
        public boolean nextKeyValue() throws IOException {
            if (current_document_ >= documents_sizes_.size()) {
                current_document_text_ = new Text();
                return false;
            }

            int document_size = documents_sizes_.get(current_document_);
            byte[] bytes = new byte[document_size];
            input_stream_.readFully(bytes, 0, document_size);

            current_document_text_ = new Text();
            Inflater decompressor = new Inflater();
            decompressor.setInput(bytes);

            byte[] decompression_buffer = new byte[MAX_BUFFER_SIZE];
            while (true) {
                try {
                    int len_readed = decompressor.inflate(decompression_buffer);
                    if (len_readed > 0) {
                        current_document_text_.append(decompression_buffer, 0, len_readed);
                    } else {
                        break;
                    }
                } catch (DataFormatException err) {
                    LOG.warn("Cannot decompress document with offset "+current_offset_);
                    break;
                }
            }
            decompressor.end();

            current_document_++;
            current_offset_ += document_size;
            return true;
        }

        @Override
        public LongWritable getCurrentKey() {
            return new LongWritable(current_offset_);
        }

        @Override
        public Text getCurrentValue() {
            return current_document_text_;
        }

        @Override
        public float getProgress() {
            return (float) current_document_ / documents_sizes_.size();
        }

        @Override
        public void close() {
            IOUtils.closeStream(input_stream_);
        }
    }

    @Override
    public DeflateRecordReader createRecordReader(InputSplit split, TaskAttemptContext context) throws IOException {
        DeflateRecordReader reader = new DeflateRecordReader();
        reader.initialize(split, context);
        return reader;
    }

    @Override
    public List<InputSplit> getSplits(JobContext context) throws IOException {
        Configuration conf = context.getConfiguration();
        List<InputSplit> splits = new ArrayList<>();

        for (FileStatus status: listStatus(context)) {
            Path file_path = status.getPath();
            Path file_index_path = file_path.suffix(".idx");
            FileSystem fs = file_path.getFileSystem(conf);
            FSDataInputStream index_file = fs.open(file_index_path);

            long split_offset = 0;
            long split_size = 0;
            ArrayList<Integer> documents_in_split_sizes = new ArrayList<>();
            while (true) {
                int size_of_document = 0;
                try {
                    size_of_document = Integer.reverseBytes(index_file.readInt());
                } catch (EOFException exc) {
                    break;
                }

                split_size += (long) size_of_document;
                documents_in_split_sizes.add(size_of_document);
                if (split_size >= getNumBytesPerSplit(conf)) {
                    splits.add(new DeflateFileSplit(file_path, split_offset, split_size,
                                                    documents_in_split_sizes));
                    split_offset += split_size;
                    split_size = 0;
                    documents_in_split_sizes.clear();
                }
            }
            if (!documents_in_split_sizes.isEmpty()) {
                splits.add(new DeflateFileSplit(file_path, split_offset, split_size,
                                                documents_in_split_sizes));
            }

            index_file.close();
        }

        return splits;
    }

    public static final String BYTES_PER_MAP_PARAMETER = "mapreduce.input.indexedgz.bytespermap";
    public static long getNumBytesPerSplit(Configuration conf) {
        return conf.getLong(BYTES_PER_MAP_PARAMETER, 32L * 1024L * 1024L);
    }
}
