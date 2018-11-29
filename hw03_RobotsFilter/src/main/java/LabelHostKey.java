import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class LabelHostKey implements WritableComparable<LabelHostKey> {
    private boolean is_robots_;
    private String host_;

    public static class Compator extends WritableComparator {
        public Compator() {
            super(LabelHostKey.class, true);
        }

        @Override
        public int compare(WritableComparable key1, WritableComparable key2) {
            return key1.compareTo(key2);
        }
    }

    public static class GroupComparator extends WritableComparator {
        public GroupComparator() {
            super(LabelHostKey.class, true);
        }

        @Override
        public int compare(WritableComparable key1, WritableComparable key2) {
            return ((LabelHostKey) key1).compareOnlyHost_((LabelHostKey) key2);
        }
    }

    public LabelHostKey() {
    }

    public LabelHostKey(boolean is_robots, String host) {
        is_robots_ = is_robots;
        host_ = host;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeBoolean(is_robots_);
        out.writeUTF(host_);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        is_robots_ = in.readBoolean();
        host_ = in.readUTF();
    }

    public int GetHash() {
        int hash = host_.hashCode();
        return (hash > 0) ? hash : (-hash);
    }

    public String GetHost() {
        return host_;
    }

    public boolean IsRobots() {
        return is_robots_;
    }

    private int compareOnlyHost_(LabelHostKey key2) {
        return host_.compareTo(key2.host_);
    }

    @Override
    public int compareTo(LabelHostKey key2) {
        int result_host_compare = compareOnlyHost_(key2);
        if (result_host_compare == 0) {
            if (is_robots_ == key2.is_robots_) { return 0; }
            if (is_robots_) { return -1; }
            else { return 1; }
        }
        return result_host_compare;
    }
}
