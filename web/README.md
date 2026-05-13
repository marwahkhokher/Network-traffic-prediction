# Web Dashboard

An interactive Streamlit dashboard that demonstrates the LSTM traffic-prediction project end-to-end.

## Pages

1. **🏠 Overview** — high-level KPIs, problem statement, architecture diagram
2. **📊 Model Comparison** — interactive comparison of all forecasting methods, including selectable OD-pair prediction traces and the LSTM's relative advantage chart
3. **🔮 Live Prediction** — pick any test slot, run the LSTM live, see actual vs predicted traffic matrices side by side, get congestion warnings
4. **🚦 Closed-Loop QoS Demo** — the killer page. Shows what each forecasting method actually delivers in operational terms: congestion events prevented, overflow volume reduced, QoS actions triggered
5. **🔬 What does AnyLogic do?** — honest discussion of AnyLogic's role in the current project (data generation only), what it *could* do (closed-loop research), and how the project is structured to enable that next step

## Run

From the project root:

```bash
pip install -r requirements.txt
streamlit run web/app.py
```

Then open http://localhost:8501.

## Prerequisites

The app reads from `results/`, so you need to have run training at least once:

```bash
python src/train.py --data data/traffic_matrix.csv --window 12 --epochs 25 --hidden 200
python scripts/make_comparison_plots.py
```

If those artifacts are missing, the app shows clear error messages telling you what to run.
