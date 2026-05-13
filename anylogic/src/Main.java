/*
 * Main.java — paste these blocks into the AnyLogic Main agent
 * (see anylogic/SETUP_GUIDE.md for where each block goes).
 *
 * The code is organised by target location:
 *   // === ADDITIONAL CLASS CODE ===   → Main's "Additional class code"
 *   // === FUNCTION: onStartup ===     → Main's "On startup" action
 *   // === FUNCTION: onSlotTick ===    → body of function onSlotTick()
 *   // === FUNCTION: generateTrafficForSlot === → body of same function
 *   // === FUNCTION: writeCsvRow ===   → body of writeCsvRow()
 *   // === FUNCTION: openCsv ===       → body of openCsv()
 *   // === FUNCTION: closeCsv ===      → body of closeCsv()
 *   // === FUNCTION: onDestroy ===     → Main's "On destroy" action
 *
 * Fields already declared as Variables/Parameters in the Main canvas
 * are referenced directly (N, intervalMin, seed, tIndex,
 * trafficMatrix, eventMultiplier, eventTtl, burstState, writer).
 */

// =====================================================================
// === ADDITIONAL CLASS CODE ===
// =====================================================================
// Paste the body below into Main's Properties → "Additional class code".
// These are helper fields and a Random used across the methods.

java.util.Random rng;
double[][] baseMatrix;
double[] nodePopulation;
double[][] nodeDistance;
final double BURST_AR = 0.6;
final double NOISE_SD = 0.08;


// =====================================================================
// === FUNCTION: onStartup ===  (Main → Properties → On startup)
// =====================================================================

rng = new java.util.Random(seed);

// Allocate state
trafficMatrix   = new double[N][N];
eventMultiplier = new double[N][N];
eventTtl        = new int[N][N];
burstState      = new double[N][N];
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        eventMultiplier[i][j] = 1.0;

// Build the gravity-model base matrix
nodePopulation = new double[N];
for (int i = 0; i < N; i++) nodePopulation[i] = 0.3 + 0.7 * rng.nextDouble();

double[][] coords = new double[N][2];
for (int i = 0; i < N; i++) {
    coords[i][0] = 100.0 * rng.nextDouble();
    coords[i][1] = 100.0 * rng.nextDouble();
}
nodeDistance = new double[N][N];
baseMatrix   = new double[N][N];
double maxBase = 0.0;
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        if (i == j) { nodeDistance[i][j] = 1.0; continue; }
        double dx = coords[i][0] - coords[j][0];
        double dy = coords[i][1] - coords[j][1];
        double d  = Math.sqrt(dx * dx + dy * dy);
        nodeDistance[i][j] = Math.max(d, 1.0);
        double b = (nodePopulation[i] * nodePopulation[j]) / Math.pow(d, 1.2);
        baseMatrix[i][j] = b;
        if (b > maxBase) maxBase = b;
    }
}
// normalise so max base = 1.0
if (maxBase > 0) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            baseMatrix[i][j] /= maxBase;
}
for (int i = 0; i < N; i++) baseMatrix[i][i] = 0.0;

// Assign IDs and populations to the node agents
for (int i = 0; i < N; i++) {
    NetworkNode nd = nodes.get(i);
    nd.nodeId = i;
    nd.population = nodePopulation[i];
}

openCsv();
traceln("Simulation started: N=" + N + ", interval=" + intervalMin + "min");


// =====================================================================
// === FUNCTION: onSlotTick ===  (no args, void)
// =====================================================================

// 1. Advance event TTLs
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        if (eventTtl[i][j] > 0) eventTtl[i][j]--;
        if (eventTtl[i][j] == 0) eventMultiplier[i][j] = 1.0;
    }
}

// 2. Possibly trigger a flash-crowd event (amplify inbound traffic to a dest)
if (rng.nextDouble() < eventProbability) {
    int dst      = rng.nextInt(N);
    int duration = 2 + rng.nextInt(6);            // 2..7 slots
    double mult  = 1.8 + 1.7 * rng.nextDouble();   // 1.8..3.5x
    for (int i = 0; i < N; i++) {
        if (i == dst) continue;
        eventMultiplier[i][dst] = mult;
        eventTtl[i][dst]        = duration;
    }
    traceln("[t=" + tIndex + "] flash-crowd event → dest=" + dst
            + "  mult=" + String.format("%.2f", mult)
            + "  dur=" + duration);
}

// 3. Compute current traffic matrix
generateTrafficForSlot();

// 4. Write to CSV
writeCsvRow();

// 5. Progress log
if (tIndex % 100 == 0) traceln("slot " + tIndex + " written");
tIndex++;


// =====================================================================
// === FUNCTION: generateTrafficForSlot ===
// =====================================================================

// Hour and day-of-week from tIndex
double totalMinutes   = tIndex * intervalMin;
double hour           = (totalMinutes / 60.0) % 24.0;
int    dayOfWeek      = ((int)(totalMinutes / (60.0 * 24.0))) % 7;

// Diurnal: cosine peak at 14:00, range ~[0.3, 1.3]
double diurnal = 0.8 + 0.5 * Math.cos((hour - 14.0) * Math.PI / 12.0);
// Weekly: 0.7 on weekend (Sat=5, Sun=6), 1.0 on weekday
double weekly  = (dayOfWeek >= 5) ? 0.7 : 1.0;

for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        if (i == j) { trafficMatrix[i][j] = 0.0; continue; }

        // AR(1) burst state update
        double innov = NOISE_SD * 0.5 * rng.nextGaussian();
        burstState[i][j] = BURST_AR * burstState[i][j] + innov;

        // Multiplicative Gaussian noise
        double noise = 1.0 + NOISE_SD * rng.nextGaussian();

        double y = baseMatrix[i][j] * diurnal * weekly * noise * eventMultiplier[i][j];
        if (burstState[i][j] > 0) y += burstState[i][j];
        if (y < 0) y = 0;
        trafficMatrix[i][j] = y;
    }
}


// =====================================================================
// === FUNCTION: openCsv ===
// =====================================================================

try {
    java.io.File f = new java.io.File("traffic_matrix.csv").getAbsoluteFile();
    writer = new java.io.PrintWriter(new java.io.FileWriter(f, false));
    // header
    StringBuilder hdr = new StringBuilder("t_index,timestamp");
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            hdr.append(",y_").append(i).append("_").append(j);
    writer.println(hdr.toString());
    writer.flush();
    traceln("opened CSV at " + f.getAbsolutePath());
} catch (Exception e) {
    traceln("CSV open failed: " + e.getMessage());
}


// =====================================================================
// === FUNCTION: writeCsvRow ===
// =====================================================================

if (writer == null) return;

// ISO timestamp based on tIndex * intervalMin minutes offset from start
long millis = (long)(tIndex * intervalMin * 60.0 * 1000.0);
java.time.LocalDateTime ts =
        java.time.LocalDateTime.of(2025, 1, 1, 0, 0).plusMinutes((long)(tIndex * intervalMin));

StringBuilder sb = new StringBuilder();
sb.append(tIndex).append(",").append(ts.toString());
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        sb.append(",").append(String.format("%.6f", trafficMatrix[i][j]));
    }
}
writer.println(sb.toString());
// flush every 50 rows to bound memory
if (tIndex % 50 == 0) writer.flush();


// =====================================================================
// === FUNCTION: closeCsv ===
// =====================================================================

if (writer != null) {
    writer.flush();
    writer.close();
    writer = null;
    traceln("closed CSV");
}


// =====================================================================
// === FUNCTION: onDestroy ===  (Main → Properties → On destroy)
// =====================================================================

closeCsv();
