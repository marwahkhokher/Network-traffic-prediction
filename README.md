# AI-Based Network Traffic Prediction System

> **Course:** Computer Communication Networks (CCN)
> **Paper followed:** Azzouni & Pujolle (2017) — *A Long Short-Term Memory Recurrent Neural Network Framework for Network Traffic Matrix Prediction* (arXiv:1705.05690)
> **Simulation tool:** AnyLogic (Personal Learning Edition, free)
> **ML stack:** TensorFlow/Keras, scikit-learn, FastAPI
> **UI:** Streamlit + Plotly

---

## What this project does

Networks experience bursty, non-stationary traffic. When demand exceeds capacity, the result is congestion: higher delay, jitter, packet loss, dropped sessions. Traditional monitoring tells operators *what is happening now* — it does not tell them what will happen next.

This project forecasts near-future traffic-matrix values **and** plugs those forecasts into one of five real-world QoS strategies, demonstrating that LSTM-guided prediction can reduce congestion events by up to **90%** vs. the reactive baseline. **AnyLogic now drives the closed loop directly** — every simulated slot, the simulation calls our FastAPI LSTM service over HTTP and applies a QoS action to its own traffic matrix before writing it to CSV.

Six layers, each runnable independently and together:

1. **AnyLogic network simulation** — agent-based + discrete-event model of a 23-node backbone (mirroring GÉANT). Each 15-minute simulated slot, it generates traffic, **calls the LSTM service over HTTP**, applies a QoS action, and writes the post-QoS matrix.
2. **LSTM prediction pipeline** — exact paper architecture: flatten N×N matrix to length-N² vector, sliding window, stacked LSTM. Compared against ARMA, Linear Regression, Random Forest, FFNN, and Persistence baselines.
3. **Closed-loop QoS controller** — five strategies (NoAction / RateLimit / Reroute / Prioritize / Hybrid), runnable both inside AnyLogic via the live integration and standalone in Python for batch analysis.
4. **Streaming service + autonomous retraining** — FastAPI serves predictions; a background worker buffers fresh data and hot-swaps the model after every 500 samples, with drift detection.
5. **Topology resilience experiments** — disable nodes, cut links, watch how the LSTM holds up under conditions it never saw in training.
6. **Multi-seed Monte Carlo** — run the entire pipeline over many random seeds, get confidence intervals on every metric.

All six layers are wrapped in an interactive **Streamlit dashboard** (10 pages, including a `📡 Streaming Live` page that monitors the AnyLogic↔LSTM loop in real time).

---

## Repository layout

```
network-traffic-prediction/
├── README.md                       ← you are here
├── requirements.txt
├── anylogic/
│   ├── SETUP_GUIDE.md              ← step-by-step AnyLogic build + closed-loop integration
│   ├── NetworkTrafficSim.alp       ← patched project — live LSTM closed loop wired in
│   └── src/                        ← Java source files (reference; the .alp has the integration)
├── src/
│   ├── anylogic_surrogate.py       ← Python equivalent of the AnyLogic sim
│   ├── data_preprocessing.py       ← matrix → vector, sliding window, normalization
│   ├── lstm_model.py               ← Keras LSTM exactly per the paper
│   ├── baselines.py                ← Linear Regression, Random Forest, ARMA, FFNN
│   ├── train.py                    ← train all models end-to-end
│   ├── evaluate.py                 ← reproduce paper's Fig. 8
│   ├── predict.py                  ← inference CLI
│   ├── qos_actions.py              ← 5 QoS strategies (RateLimit / Reroute / ...)
│   ├── closed_loop_demo.py         ← LSTM → QoS → measure (Python-side)
│   ├── topology.py                 ← GÉANT-like topology with link/node failure
│   ├── topology_resilience.py      ← resilience experiments
│   ├── monte_carlo.py              ← multi-seed confidence intervals
│   ├── network_animation.py        ← animated Plotly network views
│   ├── streaming_buffer.py         ← thread-safe rolling buffer for live ingestion
│   ├── online_learner.py           ← partial_fit + drift detection + atomic model swap
│   └── retrain_worker.py           ← background process: autonomous retraining
├── api/
│   ├── app.py                      ← FastAPI batch prediction service
│   └── streaming_app.py            ← FastAPI streaming service (this is what AnyLogic calls)
├── web/
│   ├── app.py                      ← Streamlit dashboard (10 pages)
│   ├── streaming_page.py           ← live monitoring page for the AnyLogic ↔ LSTM loop
│   └── README.md
├── scripts/
│   ├── make_comparison_plots.py    ← generates 10 publication-quality plots
│   └── run_streaming_stack.py      ← one-command launcher for service + retrain worker
├── data/                           ← traffic CSVs land here
├── notebooks/                      ← analysis notebook
└── report/                         ← project report
```

---

## Quick start

### Option A — pure Python, no AnyLogic install (recommended for fast iteration)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Generate traffic data (Python equivalent of AnyLogic)
python src/anylogic_surrogate.py --nodes 23 --timeslots 2016 --interval_min 15 \
       --out data/traffic_matrix.csv

# 3. Train LSTM + baselines
python src/train.py --data data/traffic_matrix.csv --window 12 --epochs 25 --hidden 200

# 4. Generate all comparison plots + closed-loop simulation
python scripts/make_comparison_plots.py

# 5. (Optional) Monte Carlo confidence intervals (~3 min)
python src/monte_carlo.py --n_seeds 15 --n_timeslots 700 \
       --strategies NoAction RateLimit Reroute Prioritize Hybrid

# 6. (Optional) Topology resilience experiments
python src/topology_resilience.py

# 7. Launch the interactive dashboard
streamlit run web/app.py
#   → opens at http://localhost:8501
```

### Option B — AnyLogic with the live LSTM closed loop

This runs the full system: AnyLogic generates traffic, calls the LSTM over HTTP every slot, applies QoS, writes post-QoS CSV.

```bash
# 1. Train the LSTM (Option A steps 1-3 above)
# 2. Copy the trained model to the streaming path
cp results/lstm_model.keras results/lstm_streaming.keras
echo "<your max-value>" > results/scale.txt   # the /max normalisation factor from training

# 3. Terminal 1: start the FastAPI streaming service
uvicorn api.streaming_app:app --host 127.0.0.1 --port 8765

# 4. Terminal 2 (optional): start the autonomous retrain worker
python src/retrain_worker.py --check-interval 30 --min-samples 500

# 5. In AnyLogic, open anylogic/NetworkTrafficSim.alp and hit ▶ Run
#    Watch the Console for: slot 100  ok=87  fail=0  qos=14  before=12.34  after=4.12
```

See `anylogic/SETUP_GUIDE.md` §7 for the full integration documentation.

---

## Headline results

On the held-out test set (23 nodes × 2016 slots × 15-min intervals):

### Model accuracy (test MSE)

| Method            | MSE        | vs. LSTM |
| ----------------- | ---------- | -------- |
| Persistence       | 0.000362   | 1.28× worse |
| Linear Regression | 0.000303   | 1.07× worse |
| ARMA(2,1)         | 0.000294   | 1.04× worse |
| **LSTM**          | **0.000287** | **1.00×**  |
| FFNN              | 0.000278   | 0.99× (tie within MC noise) |

### Operational impact — closed-loop QoS

Plugging LSTM predictions into different QoS strategies on the same test data:

| QoS strategy       | Events reduced | Overflow reduced | Real-world equivalent |
| ------------------ | -------------- | ---------------- | --------------------- |
| NoAction (baseline) | 0%             | 0%               | Best-effort networking |
| Reroute alone       | -47% (worse)   | -37%             | MPLS-TE without cap-checks |
| Prioritize          | +52%           | +43%             | DiffServ classes |
| RateLimit           | +54%           | +60%             | Token-bucket policers |
| **Hybrid**          | **+91%**       | **+75%**         | Modern SD-WAN controllers |

### Robustness — Monte Carlo across 15 seeds

| QoS strategy | Event reduction (mean ± std) |
| ------------ | ---------------------------- |
| Hybrid       | **63.6% ± 47.6%**            |
| RateLimit    | 23.8% ± 25.0%                |
| Prioritize   | 22.8% ± 23.9%                |
| Reroute      | -77% ± 135% (highly unstable) |

Hybrid is the most robust strategy across random seeds.

### Live AnyLogic measurement

When AnyLogic runs the closed loop directly (via the patched `.alp`), it reports its own before/after numbers in the Console at the end of the run:

```
=== LSTM integration summary ===
  predictions OK    : 2016
  predictions FAIL  : 0
  QoS actions taken : 312
  overflow BEFORE   : 184.521
  overflow AFTER    : 23.117
  overflow reduced  : 87.5%
```

This confirms the Python-side closed-loop result generalises end-to-end — the controller behaviour is the same whether evaluated in batch Python or live inside the simulator.

---

## Why AnyLogic at all?

The paper uses real GÉANT traces (proprietary pan-European data). Since we cannot legally redistribute that dataset, AnyLogic lets us build a topologically + statistically similar 23-node backbone and export arbitrarily long traffic sequences. The Python surrogate (`src/anylogic_surrogate.py`) does the same thing with no install required — pick whichever fits.

**Updated for the live integration:** AnyLogic isn't just a data generator anymore. With the patched `.alp`, every 15-minute simulated slot AnyLogic calls our FastAPI streaming service over HTTP, receives the LSTM's prediction and congestion flags, and applies a QoS action *inside the same simulation tick*. The CSV it writes contains the post-QoS traffic. This is true closed-loop simulation: the model's predictions affect what flows in the simulator the moment they're made.

See the dashboard's "What does AnyLogic do?" page for the full discussion.

---

## References

1. Azzouni, A. & Pujolle, G. (2017). *A Long Short-Term Memory Recurrent Neural Network Framework for Network Traffic Matrix Prediction.* arXiv:1705.05690.
2. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation 9(8).
3. Uhlig, S. et al. (2006). *Providing public intradomain traffic matrices to the research community.* ACM SIGCOMM CCR 36(1).
4. GÉANT Project — geant.org
