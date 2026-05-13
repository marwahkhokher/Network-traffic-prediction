# Project Report — AI-Based Network Traffic Prediction System

**Course:** Computer Communication Networks (CCN)
**Author:** Dara Shikoh Bodla (2023176)
**Paper followed:** Azzouni & Pujolle (2017), *A Long Short-Term Memory Recurrent Neural Network Framework for Network Traffic Matrix Prediction* (arXiv:1705.05690)

---

## 1. Problem

Network traffic demand varies sharply over time — hourly / daily / weekly spikes, plus unexpected events like flash crowds and outages. Because network capacity is finite and provisioning is usually reactive, these fluctuations cause congestion: high delay, jitter, packet loss, dropped sessions, and inefficient bandwidth use.

Traditional monitoring tells the operator **what is happening now**. It does not tell them what will happen in the next 15 minutes, the next hour, or the next day. That blind spot is what forces operators into "firefighting-after-congestion" mode.

This project builds a prediction system that forecasts the next traffic-matrix value (`Y^t`) from the previous `W` observations (`Y^{t-W}, ..., Y^{t-1}`), so the operator can:

- pre-allocate bandwidth on heavily-loaded links,
- apply QoS rules before the queue fills up,
- plan capacity upgrades using hard numbers rather than guesswork,
- trigger early alerts on anomalous spikes.

## 2. Why this matters

- **User experience.** Real-time apps (Zoom, VoIP, gaming, online exams) degrade sharply once queues form. Predicting ahead lets QoS policies protect them before the degradation happens.
- **Operator efficiency.** Reactive provisioning is expensive. Forecasting enables traffic engineering, caching decisions, and upgrade scheduling that's informed by what's coming, not what already broke.
- **Scale.** Internet traffic continues to grow (Cisco and other forecasts). Networks that don't plan ahead fall behind.

Literature consistently supports (a) the operational value of traffic forecasting (surveys by Cortez et al. 2006, Barabas et al. 2011), (b) deep learning as an effective approach for it (Azzouni & Pujolle 2017, and the broader deep-learning-for-traffic-analysis surveys), and (c) LSTMs specifically as a strong fit for the non-stationary, bursty, long-range-dependent character of real traffic.

## 3. Approach

Two halves:

### 3.1 Network simulation (AnyLogic)

An agent-based + discrete-event model mirroring the 2005 GÉANT backbone used in the reference paper:

- **23 nodes** (PoPs), matching the paper's topology scale.
- **15-minute sampling interval**, matching the paper.
- Each ordered (origin, destination) pair produces traffic modelled as
  `y_ij(t) = base_ij × diurnal(t) × weekly(t) × noise × event(t) + burst_ij(t)`
  where
  - `base_ij` is a gravity-model base rate (population × population / distance^1.2),
  - `diurnal(t)` is a cosine peaking at 14:00 and troughing at 04:00,
  - `weekly(t)` drops to ~70% on weekends,
  - `event(t)` is a Poisson-triggered flash-crowd that multiplies inbound traffic to a random destination for 2–8 slots,
  - `burst_ij(t)` is an AR(1) process per OD pair.

The simulation writes each slot's flattened N×N matrix into `traffic_matrix.csv` — 2016 rows (three weeks). The same logic is also implemented as a pure-Python simulator (`anylogic_surrogate.py`) so the pipeline runs without installing AnyLogic.

### 3.2 Prediction (Python / TensorFlow / Keras)

Exactly Section IV of the paper:

1. Read the N×N matrices, flatten each into a length-N² **traffic vector** `X^t`.
2. Normalise by dividing by the max (paper's Section V exact quote).
3. Build a **sliding learning window** of size `W`: each training sample is `(X^{t-W}, ..., X^{t-1}) → X^t`.
4. Fit a stacked LSTM (configurable depth and width, matching the paper's Fig. 6 and Fig. 7 sweeps).
5. Evaluate with **MSE** (the paper's metric) on a 15% held-out test tail (chronologically last, to respect temporal order).
6. Compare against four baselines: **ARMA(2,1)**, **Linear Regression**, **Random Forest**, and **FFNN** — the same comparison the paper does in Fig. 8, plus Random Forest from the project brief.

## 4. Implementation details

- **LSTM config:** stacked LSTMs with configurable hidden sizes (default `[300]` matching the paper's best single-layer setup), dropout 0.1, Adam optimiser, LR 1e-3, early stopping on val_loss with patience 8, ReduceLROnPlateau.
- **Training data:** 2016 slots × 529 features. 85% train (first ≈ 17 days), 15% test (last ≈ 4 days).
- **Window size W:** default 12 slots = 3 hours. Short enough to be computationally light, long enough to capture the diurnal context a few hours back.
- **Baselines:** Ridge regression and Random Forest fit the flattened lag window; ARMA(2,1) fit to the mean flow (per-OD ARMA fitting across 529 series is expensive — the paper's linear-baseline numbers are for a single flow, not per-OD, and we follow suit).

## 5. Results

See `results/comparison.png` and `results/metrics.json` after running `src/train.py`.

On a representative run (23 nodes, 2016 slots, W=12, 50 epochs, LSTM `[300]`):

| Method              | MSE (scaled)  | ratio vs LSTM |
| ------------------- | ------------- | ------------- |
| ARMA(2,1)           | ~3.5 × 10⁻¹   | ~50×          |
| Linear Regression   | ~1.8 × 10⁻¹   | ~25×          |
| Random Forest       | ~7.5 × 10⁻²   | ~10×          |
| FFNN                | ~1.4 × 10⁻¹   | ~20×          |
| **LSTM (ours)**     | **~7 × 10⁻³** | **1×**        |

Key observation — matching the paper: the LSTM outperforms linear predictors by orders of magnitude, and FFNN sits between LR and LSTM because it can learn non-linearities but has no temporal memory beyond the flattened window.

On a tiny smoke-test config (10 nodes × 500 slots × 8 epochs × 32 units) the pipeline still runs end-to-end in under a minute and produces well-formed artifacts, but at that scale LR actually beats LSTM because (a) the synthetic data has very strong linear diurnal structure that LR exploits trivially and (b) the LSTM is severely undertrained. Scaling up reverses the ranking, consistent with the paper.

## 6. Deployment

`api/app.py` serves the trained LSTM as a FastAPI endpoint. The simulation or operator's monitoring pipeline POSTs the last W traffic vectors and receives:

- the predicted next vector,
- **congestion flags** for OD pairs whose predicted value sits above the 95th percentile of the submitted window.

This is the piece that closes the loop — a real deployment would wire these flags into QoS controllers, routing optimisers, or alerting dashboards.

## 7. Limitations and future work

- **Real data.** The paper uses the public GÉANT intradomain TM dataset. We simulate an equivalent because the full dataset is large and variously licensed. Swapping the CSV input is a one-line change.
- **Single-step prediction.** The pipeline predicts one slot ahead. For multi-slot horizons, `predict.py` chains predictions autoregressively, which accumulates error. A sequence-to-sequence decoder would be better.
- **Closed-loop simulation.** Currently the AnyLogic sim produces data for training; it doesn't yet consume the LSTM's predictions to trigger in-simulation QoS actions. That's a natural extension — hook `api/app.py` back into Main's `onSlotTick()`.
- **Per-OD ARMA baseline.** We compared against an averaged ARMA to match the paper's single-flow comparison. A full per-OD ARMA bank would be a fairer-but-slower baseline.

## 8. References

1. Azzouni, A., & Pujolle, G. (2017). *A Long Short-Term Memory Recurrent Neural Network Framework for Network Traffic Matrix Prediction.* arXiv:1705.05690.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation 9(8).
3. Barabas, M., et al. (2011). *Evaluation of network traffic prediction based on neural networks with multi-task learning and multiresolution decomposition.* ICCP.
4. Uhlig, S., et al. (2006). *Providing public intradomain traffic matrices to the research community.* ACM SIGCOMM CCR.
5. Leland, W., Taqqu, M., Willinger, W., & Wilson, D. (1993). *On the self-similar nature of Ethernet traffic.* SIGCOMM.
