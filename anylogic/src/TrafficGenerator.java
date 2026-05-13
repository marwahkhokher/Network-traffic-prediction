/*
 * TrafficGenerator.java — standalone Java helper class.
 *
 * This is a REFERENCE implementation that mirrors the logic embedded
 * in Main.java's onSlotTick() / generateTrafficForSlot() / etc.
 *
 * Two use cases:
 *   1. You can ALSO create this as a regular Java class inside the
 *      AnyLogic project (Projects panel → right-click model → New → Java Class)
 *      and then Main can call it via
 *            tgen = new TrafficGenerator(N, seed);
 *            tgen.nextSlot(tIndex, intervalMin);
 *      That makes Main shorter and keeps generation logic unit-testable.
 *
 *   2. You can compile and run this class outside AnyLogic, purely with
 *      `javac + java`, to produce a CSV identical to what the simulation
 *      produces. Handy for CI and for grading environments without AnyLogic.
 *
 * Compile & run standalone:
 *      javac TrafficGenerator.java
 *      java  TrafficGenerator 23 2016 15 traffic_matrix.csv
 */

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.util.Random;

public class TrafficGenerator {

    // ---- configuration ----
    private final int    N;
    private final long   seed;
    private final double eventProbability;
    private static final double BURST_AR  = 0.6;
    private static final double NOISE_SD  = 0.08;

    // ---- state ----
    private final Random    rng;
    private final double[][] baseMatrix;
    private final double[][] trafficMatrix;
    private final double[][] eventMultiplier;
    private final int[][]    eventTtl;
    private final double[][] burstState;

    public TrafficGenerator(int N, long seed) {
        this(N, seed, 0.01);
    }

    public TrafficGenerator(int N, long seed, double eventProbability) {
        this.N                = N;
        this.seed             = seed;
        this.eventProbability = eventProbability;
        this.rng              = new Random(seed);

        this.trafficMatrix    = new double[N][N];
        this.eventMultiplier  = new double[N][N];
        this.eventTtl         = new int[N][N];
        this.burstState       = new double[N][N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                eventMultiplier[i][j] = 1.0;

        // --- gravity-model base matrix ---
        double[] pop = new double[N];
        for (int i = 0; i < N; i++) pop[i] = 0.3 + 0.7 * rng.nextDouble();
        double[][] coords = new double[N][2];
        for (int i = 0; i < N; i++) {
            coords[i][0] = 100.0 * rng.nextDouble();
            coords[i][1] = 100.0 * rng.nextDouble();
        }
        baseMatrix = new double[N][N];
        double maxBase = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                double dx = coords[i][0] - coords[j][0];
                double dy = coords[i][1] - coords[j][1];
                double d  = Math.max(Math.sqrt(dx * dx + dy * dy), 1.0);
                double b  = (pop[i] * pop[j]) / Math.pow(d, 1.2);
                baseMatrix[i][j] = b;
                if (b > maxBase) maxBase = b;
            }
        }
        if (maxBase > 0) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    baseMatrix[i][j] /= maxBase;
        }
    }

    /** Compute the next slot's traffic matrix in place. */
    public double[][] nextSlot(int tIndex, double intervalMin) {
        // event decay
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (eventTtl[i][j] > 0) eventTtl[i][j]--;
                if (eventTtl[i][j] == 0) eventMultiplier[i][j] = 1.0;
            }
        }
        // event trigger
        if (rng.nextDouble() < eventProbability) {
            int    dst      = rng.nextInt(N);
            int    duration = 2 + rng.nextInt(6);
            double mult     = 1.8 + 1.7 * rng.nextDouble();
            for (int i = 0; i < N; i++) {
                if (i == dst) continue;
                eventMultiplier[i][dst] = mult;
                eventTtl[i][dst]        = duration;
            }
        }

        double totalMinutes = tIndex * intervalMin;
        double hour         = (totalMinutes / 60.0) % 24.0;
        int    dayOfWeek    = ((int)(totalMinutes / (60.0 * 24.0))) % 7;
        double diurnal      = 0.8 + 0.5 * Math.cos((hour - 14.0) * Math.PI / 12.0);
        double weekly       = (dayOfWeek >= 5) ? 0.7 : 1.0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) { trafficMatrix[i][j] = 0.0; continue; }
                double innov = NOISE_SD * 0.5 * rng.nextGaussian();
                burstState[i][j] = BURST_AR * burstState[i][j] + innov;
                double noise = 1.0 + NOISE_SD * rng.nextGaussian();
                double y = baseMatrix[i][j] * diurnal * weekly * noise * eventMultiplier[i][j];
                if (burstState[i][j] > 0) y += burstState[i][j];
                trafficMatrix[i][j] = Math.max(0, y);
            }
        }
        return trafficMatrix;
    }

    public double[][] getTrafficMatrix() { return trafficMatrix; }
    public int        getN()             { return N; }

    // ------------------------------------------------------------------
    // Standalone CLI:
    //   java TrafficGenerator <N> <timeslots> <intervalMin> <out.csv>
    // ------------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        int    N           = args.length > 0 ? Integer.parseInt(args[0]) : 23;
        int    timeslots   = args.length > 1 ? Integer.parseInt(args[1]) : 2016;
        double intervalMin = args.length > 2 ? Double.parseDouble(args[2]) : 15.0;
        String outPath     = args.length > 3 ? args[3] : "traffic_matrix.csv";

        TrafficGenerator tg = new TrafficGenerator(N, 7L);

        try (PrintWriter w = new PrintWriter(new FileWriter(new File(outPath)))) {
            StringBuilder hdr = new StringBuilder("t_index,timestamp");
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    hdr.append(",y_").append(i).append("_").append(j);
            w.println(hdr);

            LocalDateTime start = LocalDateTime.of(2025, 1, 1, 0, 0);
            for (int t = 0; t < timeslots; t++) {
                double[][] m = tg.nextSlot(t, intervalMin);
                StringBuilder sb = new StringBuilder();
                sb.append(t).append(",").append(start.plusMinutes((long)(t * intervalMin)));
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        sb.append(",").append(String.format("%.6f", m[i][j]));
                w.println(sb);
            }
        }
        System.out.println("wrote " + outPath + "  (" + timeslots + " rows)");
    }
}
