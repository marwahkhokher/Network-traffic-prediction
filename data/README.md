# data/

## What lives here

- `traffic_matrix.csv` — the main training file. One row per 15-min slot, with the N×N traffic matrix flattened row-major into columns `y_0_0, y_0_1, ..., y_{N-1,N-1}`. Plus `t_index` and `timestamp` columns.

## How to regenerate

### From Python (fast, no AnyLogic needed):

```bash
python src/anylogic_surrogate.py --nodes 23 --timeslots 2016 --interval_min 15 \
    --out data/traffic_matrix.csv
```

### From AnyLogic (authoritative):

1. Build the model per `anylogic/SETUP_GUIDE.md`
2. Run the simulation — it writes `traffic_matrix.csv` into the AnyLogic project root
3. Copy it here: `cp /path/to/anylogic_workspace/NetworkTrafficSim/traffic_matrix.csv data/`

### From the compiled Java helper:

```bash
cd anylogic/src
javac TrafficGenerator.java
java TrafficGenerator 23 2016 15 ../../data/traffic_matrix.csv
```

All three methods produce the same CSV format.

## Format reference

```
t_index,timestamp,y_0_0,y_0_1,...,y_22_22
0,2025-01-01 00:00:00,0.000000,0.123456,...,0.000000
1,2025-01-01 00:15:00,...
...
```

- `t_index` : 0-based slot counter
- `timestamp` : ISO datetime (useful for plots; the ML pipeline doesn't depend on it)
- `y_i_j` : traffic volume from node i to node j in this slot (units: arbitrary — the ML pipeline normalises by dividing by the max, following the paper)
- Diagonal entries `y_i_i` are always 0 (no self-traffic)

## Why 2016 slots?

2016 slots × 15 min = 21 days = 3 weeks. That gives the LSTM enough diurnal + weekly structure to learn, leaves room for a 15% held-out test set (≈ 3 days), and keeps training time under a few minutes on CPU. The paper uses 309 slots from GÉANT, but with N=23 and only 309 slots you can't reliably train a 300-unit LSTM — we err on the side of more data.
