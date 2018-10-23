import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class HostQueryKey implements WritableComparable<HostQueryKey> {
    private String host_;
    private String query_;

    public static class Compator extends WritableComparator {
        public Compator() {
            super(HostQueryKey.class, true);
        }

        @Override
        public int compare(WritableComparable key1, WritableComparable key2) {
            return key1.compareTo(key2);
        }
    }

    public static class GroupComparator extends WritableComparator {
        public GroupComparator() {
            super(HostQueryKey.class, true);
        }

        @Override
        public int compare(WritableComparable key1, WritableComparable key2) {
            return ((HostQueryKey) key1).compareOnlyHost_((HostQueryKey) key2);
        }
    }

    public HostQueryKey() {
    }

    public HostQueryKey(String host, String query) {
        host_ = host;
        query_ = query;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeUTF(host_);
        out.writeUTF(query_);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        host_ = in.readUTF();
        query_ = in.readUTF();
    }

    public String getHostName() {
        return host_;
    }

    @Override
    public String toString() {
        return host_+"\t"+query_;
    }

    public String getQuery() {
        return query_;
    }

    @Override
    public int hashCode() {
        return host_.hashCode();
    }

    private int compareOnlyHost_(HostQueryKey key2) {
        return host_.compareTo(key2.host_);
    }

    @Override
    public int compareTo(HostQueryKey key2) {
        int result_host_compare = compareOnlyHost_(key2);
        if (result_host_compare == 0) {
            return query_.compareTo(key2.query_);
        }
        return result_host_compare;
    }
}
