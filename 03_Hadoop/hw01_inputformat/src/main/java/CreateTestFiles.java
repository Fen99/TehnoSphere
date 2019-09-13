import java.io.*;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

public class CreateTestFiles {
    private static void compressFiles(String result_name, String[] strs)
            throws FileNotFoundException, IOException, Exception {
        FileOutputStream out = new FileOutputStream(result_name);
        FileOutputStream out_index = new FileOutputStream(result_name+".idx");

        int len_compressed = 0;
        for (String str: strs) {
            byte[] bytes_str = str.getBytes("UTF-8");

            Deflater compressor = new Deflater();
            compressor.setInput(bytes_str);
            compressor.finish();
            byte[] compressed = new byte[100];
            int compressed_len = compressor.deflate(compressed);

            System.out.print(compressed_len+"\n"+Integer.reverseBytes(compressed_len)+"\n");

            out.write(compressed, 0, compressed_len);
            ByteBuffer buff = ByteBuffer.allocate(4);
            buff.putInt(Integer.reverseBytes(compressed_len));
            out_index.write(buff.array());

            Inflater decomp = new Inflater();
            decomp.setInput(compressed, 0, compressed_len);
            decomp.finished();
            byte[] bytes_decomp = new byte[100];
            int len = decomp.inflate(bytes_decomp);
            System.out.print(len+"|"+new String(bytes_decomp, 0, len, "UTF-8") +"\n");
            System.out.print(Arrays.toString(compressed)+"\n\n");
        }

        out.close();
        out_index.close();
    }

    public static void main(String[] args) throws Exception {
        String[] f1 = {"abc", "xqwerty\nabc\ndef\ndef", "dfsdf", "2389\nsdfjfffk"};
        String[] f2 = {"mnbvpoi\nabc\ndef", "asdf", "dsdf", "wefo\ndfj"};
        compressFiles("test1.pkz", f1);
        compressFiles("test2.pkz", f2);
    }
}
