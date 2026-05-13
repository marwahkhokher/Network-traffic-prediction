/*
 * TrafficExporter.java — reference Java helper class.
 *
 * Small utility that writes a single traffic matrix row to an
 * open PrintWriter in the exact CSV format the ML pipeline expects:
 *
 *     t_index,timestamp,y_0_0,y_0_1,...,y_{N-1,N-1}
 *
 * Not strictly required inside AnyLogic (Main.java's writeCsvRow()
 * already does this inline), but handy if you prefer to split
 * I/O out of Main for cleanliness.
 */

import java.io.PrintWriter;
import java.time.LocalDateTime;

public class TrafficExporter {

    private final PrintWriter writer;
    private final int         N;
    private final LocalDateTime startTime;
    private final double      intervalMin;
    private long              rowsWritten = 0;

    public TrafficExporter(PrintWriter writer, int N,
                           LocalDateTime startTime, double intervalMin) {
        this.writer      = writer;
        this.N           = N;
        this.startTime   = startTime;
        this.intervalMin = intervalMin;
        writeHeader();
    }

    private void writeHeader() {
        StringBuilder hdr = new StringBuilder("t_index,timestamp");
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                hdr.append(",y_").append(i).append("_").append(j);
        writer.println(hdr);
        writer.flush();
    }

    public void writeRow(int tIndex, double[][] matrix) {
        if (matrix.length != N || matrix[0].length != N)
            throw new IllegalArgumentException("matrix must be " + N + "x" + N);

        LocalDateTime ts = startTime.plusMinutes((long)(tIndex * intervalMin));
        StringBuilder sb = new StringBuilder();
        sb.append(tIndex).append(",").append(ts);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                sb.append(",").append(String.format("%.6f", matrix[i][j]));
        writer.println(sb);

        rowsWritten++;
        if (rowsWritten % 50 == 0) writer.flush();
    }

    public void close() {
        writer.flush();
        writer.close();
    }

    public long getRowsWritten() { return rowsWritten; }
}
