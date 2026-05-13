# AnyLogic Setup Guide

This guide explains how to build (or run) the **NetworkTrafficSim** model in AnyLogic Personal Learning Edition (free).

You have two paths:

1. **Fast path:** Open the prebuilt `NetworkTrafficSim.alp` — if it opens in your AnyLogic version, you're done.
2. **Reliable path:** Build the model yourself in ~15 minutes by following the steps below. This always works because you paste the Java source files from `src/` directly into AnyLogic.

---

## 0. Prerequisites

- Download **AnyLogic Personal Learning Edition** (free) from <https://www.anylogic.com/downloads/>.
- Install and open it. Accept the default workspace.

---

## 1. Create the project

1. `File → New → Model…`
2. Model name: `NetworkTrafficSim`, Java package: `networktrafficsim`. Finish.

You should now see an empty `Main` agent canvas.

---

## 2. Create the `NetworkNode` agent type

1. In the Projects panel (left), right-click the model → `New → Agent…`.
2. Choose **"Agent type"**, name it `NetworkNode`. Click Next.
3. Choose "I want to create the agent population later". Click Next → Finish.
4. Open `NetworkNode` by double-clicking. In the **Properties panel** (right) scroll to **"Additional class code"** and paste the entire contents of `anylogic/src/NetworkNode.java` there (the body between the outermost braces — AnyLogic wraps it in a class declaration for you).

What this agent holds:
- `int nodeId` — the node's index (0..N-1)
- `double population` — its relative size in the gravity model
- `double[] outgoingRates` — length-N array, rate to each destination
- methods `recomputeRates(int hour, int dayOfWeek, boolean inEvent)` and `recordInboundTraffic(int origin, double volume)`.

---

## 3. Create the `Main` agent — the network

Main already exists. Open it and:

1. **Parameters** (drag from palette → *Agent* → *Parameter*):
   - `N` : int, default `23`
   - `intervalMin` : double, default `15.0`
   - `simulationDays` : double, default `21`
   - `eventProbability` : double, default `0.01`
   - `seed` : long, default `7`

2. **Populations** (drag → *Agent* → *Agent Population*):
   - Name: `nodes`; Agent type: `NetworkNode`; Initial number of agents: `N`.

3. **Variables**:
   - `double[][] trafficMatrix` — the current N×N matrix (initial value `new double[N][N]`)
   - `double[][] eventMultiplier`
   - `int[][] eventTtl`
   - `double[][] burstState`
   - `int tIndex` — slot counter (default 0)
   - `java.io.PrintWriter writer` — CSV output

4. **Events** (drag → *Agent* → *Event*):
   - Name: `slotTick`
   - Trigger type: **Timeout**
   - Mode: **Cyclic**, recurrence time: `intervalMin` minutes
   - Action: `onSlotTick();`

5. **Functions** (drag → *Agent* → *Function*):
   - `onSlotTick` (no args, no return): paste the body from `Main.java#onSlotTick`
   - `generateTrafficForSlot` : body from `Main.java#generateTrafficForSlot`
   - `writeCsvRow` : body from `Main.java#writeCsvRow`
   - `openCsv` : body from `Main.java#openCsv`
   - `closeCsv` : body from `Main.java#closeCsv`

6. **On startup** (select Main, Properties → *On startup*): paste `Main.java#onStartup`.
7. **On destroy** (Properties → *On destroy*): paste `Main.java#onDestroy` (closes the CSV).

---

## 4. Configure the simulation experiment

1. In the Projects panel, open `Simulation: Main`.
2. Set **Model time units** to `minutes`.
3. Set **Stop time** to `simulationDays * 24 * 60` (in minutes). You can type the expression directly.
4. Execution mode: **Virtual time (as fast as possible)** — you want the CSV produced quickly, not real time.

---

## 5. Run the model

Press **Run** (green ▶). You should see:

- Console: `opened CSV at /absolute/path/traffic_matrix.csv`
- Console: one log line per 100 slots (optional)
- After ~30 s – 2 min the simulation stops and the CSV is fully written.

---

## 6. Move the output to the ML pipeline

The CSV is written to the AnyLogic project's root (or wherever your OS resolves a relative path — check the console message for the absolute path).

```bash
cp /path/to/AnyLogic/workspace/NetworkTrafficSim/traffic_matrix.csv \
   ~/network-traffic-prediction/data/traffic_matrix.csv
```

Then run the ML pipeline:

```bash
cd ~/network-traffic-prediction
python src/train.py --data data/traffic_matrix.csv --window 12 --epochs 50
```

---

## Troubleshooting

- **"Cannot find symbol"** when pasting Java: make sure you pasted into *Additional class code* (for fields/helpers) or inside a *Function* body (for method code). Don't paste the `class Main {` wrapper — AnyLogic provides it.
- **CSV is empty**: the event probably never fired. Check the event's recurrence is in minutes and the Main time unit is minutes.
- **Model runs forever**: did you set Stop time correctly? Should be `simulationDays * 24 * 60` minutes.
- **Wrong number of columns**: check N matches between the model and what `data_preprocessing.py` expects — the loader auto-detects N from column count, so any square N works.

---

## Why we model it this way

The paper uses the GÉANT 2005 TM dataset — 23 nodes, 529 OD pairs, 15-min slots. We replicate:

- **Topology scale:** `N = 23`.
- **Sampling interval:** 15 min.
- **Matrix structure:** flattened row-major into a 529-length vector on the ML side.
- **Realistic temporal structure:** diurnal + weekly + event-driven components, plus AR(1) burstiness per OD pair (the paper emphasises that traffic has "self-similarity, multiscalarity, long-range dependence and a highly nonlinear nature" — the surrogate AR(1) burst term captures short-range correlation, the diurnal/weekly terms capture long-range periodicity, and the event mechanism captures non-Gaussian spikes).

The simulation is intentionally simple enough that you can understand every line of it, while rich enough that the LSTM will beat linear baselines by the same order of magnitude the paper reports.

---

## 7. Live closed-loop integration (LSTM ↔ AnyLogic via FastAPI)

This is the upgrade that turns AnyLogic from a data exporter into the centre of a closed-loop control system. Each simulated slot, AnyLogic calls the trained LSTM over HTTP, applies a QoS action to the matrix it just generated, and writes the **post-QoS** matrix to CSV.

If you opened the prebuilt `NetworkTrafficSim.alp` shipped in this folder, **the integration is already wired in** — skip to §7.3 and just start the service.

### 7.1 What got added inside the model

If you're building from scratch (Section 3 above), here's what the integration needs on top of the original model.

**Three new parameters** (Main → Properties → Parameters):
- `enableLstmCalls : boolean = true` — master switch
- `lstmServiceUrl : String = "http://127.0.0.1:8765"`
- `qosStrategy : String = "Hybrid"` — one of `NoAction` / `RateLimit` / `Reroute` / `Hybrid`

**New fields in Main's *Additional class code*** (paste below the existing `BURST_AR` / `NOISE_SD` lines):
```java
java.net.http.HttpClient httpClient;
java.time.Duration       httpTimeout = java.time.Duration.ofSeconds(2);
double[][] rawTrafficMatrix;
java.util.List<Integer> overloadedDests = new java.util.ArrayList<Integer>();
int    lstmCallsOk     = 0;
int    lstmCallsFailed = 0;
int    qosActionsTotal = 0;
int    congestionEventsAvoided = 0;
double overflowBefore  = 0.0;
double overflowAfter   = 0.0;
```

**Additions to *On startup*** (append before/after `openCsv()`):
```java
rawTrafficMatrix = new double[N][N];
httpClient = java.net.http.HttpClient.newBuilder()
        .version(java.net.http.HttpClient.Version.HTTP_1_1)  // uvicorn doesn't speak h2c
        .connectTimeout(java.time.Duration.ofSeconds(2))
        .build();
```

**Four new functions** — drag → Function from the palette and paste each body. The full bodies live inside the patched `NetworkTrafficSim.alp` and the `INTEGRATION_GUIDE.md` at the project root; the names are:
- `callPredictionService()` — void
- `applyQosAction()` — void
- `nodeCapacity(int j)` — returns `double`
- `matchingBracket(String s, int openIdx)` — returns `int`

**Modified `onSlotTick`** — after `generateTrafficForSlot()` add steps 3b through 8 from the patched file: snapshot raw matrix, call service, measure pre-QoS overflow, apply QoS, measure post-QoS overflow, write CSV.

The patched `NetworkTrafficSim.alp` in this folder contains all of the above already — recommended path is to just open it.

### 7.2 The control loop, step by step

Per tick (15 simulated minutes):

```
generateTrafficForSlot()      → fills `trafficMatrix`
copy → rawTrafficMatrix       → keep the pre-QoS twin
callPredictionService()       → POST /ingest, parse congestion_flags
overflow accounting (pre-QoS)
applyQosAction()              → modifies `trafficMatrix` in place
overflow accounting (post-QoS)
writeCsvRow()                 → writes the modified matrix
```

The CSV at the end of the run therefore contains the **post-QoS** traffic — what actually flowed after the controller intervened. To produce a before/after comparison, run the simulation twice with `enableLstmCalls = false` then `true` and compare the CSVs in the dashboard.

### 7.3 Running the integration end-to-end

**Terminal 1** — start the FastAPI streaming service:
```bash
cd path/to/network-traffic-prediction
uvicorn api.streaming_app:app --host 127.0.0.1 --port 8765
```
Wait for `Application startup complete.`. Sanity-check at <http://127.0.0.1:8765/model_info>.

**Terminal 2 (optional)** — start the autonomous retraining worker:
```bash
python src/retrain_worker.py --check-interval 30 --min-samples 500
```

**AnyLogic** — open the patched `.alp`, verify the three parameters on Main, hit ▶ Run. The Console shows live trace every 100 slots:
```
slot 100  ok=87  fail=0  qos=14  before=12.34  after=4.12
```

End-of-run summary:
```
=== LSTM integration summary ===
  predictions OK    : 2016
  predictions FAIL  : 0
  QoS actions taken : 312
  overflow BEFORE   : 184.521
  overflow AFTER    : 23.117
  overflow reduced  : 87.5%
```

### 7.4 Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Connection refused` in trace | Service not running | Start uvicorn on port 8765 |
| `HTTP 422` on every call | HTTP/2 upgrade misnegotiation | Ensure `httpClient` is built with `.version(HttpClient.Version.HTTP_1_1)` |
| `HTTP 503` from service | No model loaded | Run `python src/train.py` first, then `cp results/lstm_model.keras results/lstm_streaming.keras` |
| All slots show `flags=0` | Buffer's congestion threshold (quantile) is too high for the current traffic levels | Run for more slots so the buffer fills, or lower the quantile in `streaming_app.py::_detect_congestion` |
| AnyLogic refuses to open patched `.alp` | XML edit corrupted | Restore the `.original` backup and re-apply |

