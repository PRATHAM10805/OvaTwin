# OvaTwin - React Digital Twin Dashboard

A modern React application connected to a live FastAPI simulation backend for:
**[`src/integration/test_digital_twin.py`](file:///c:/Users/priya/OneDrive/Desktop/PCOS/src/integration/test_digital_twin.py)**

---

## Architecture Overview

```text
       ┌────────────────────────┐
       │   React App (Vite)     │  <-- Interactive Dashboard, Canvas charts, Table filters
       │ http://localhost:5173  │
       └───────────▲────────────┘
                   │  JSON (fetch)
                   │  POST /api/run-simulation
       ┌───────────▼────────────┐
       │   FastAPI Backend      │  <-- src/api/server.py
       │ http://localhost:8000  │
       └───────────▲────────────┘
                   │
                   │  Executes
       ┌───────────▼────────────┐
       │   PatientDigitalTwin   │  <-- RoadRunner CVODE, SBML BIOMD0000000494, 120d simulation
       └────────────────────────┘
```

---

## Quick Start (How to Run)

### 1. Start the Live FastAPI Backend
In your project root (`c:\Users\priya\OneDrive\Desktop\PCOS`):
```powershell
uvicorn src.api.server:app --port 8000 --reload
```
API docs available at: `http://localhost:8000/docs`

### 2. Start the React Frontend
In the `frontend/` directory:
```powershell
cd frontend
npm run dev
```
Open: `http://localhost:5173` in your browser.

---

## Features

1. **Live On-Demand Simulation:**
   - Modify the parameter (`p1`, `p17`, `p49`, `p51`) and perturbation percentage (`+0.1%`, `+5%`, `-20%`).
   - Click **"Run Live Simulation"** &mdash; the React app sends the request to the FastAPI backend, runs RoadRunner ODE simulation in real-time, and streams updated trajectories and 45-species metrics directly into the UI.

2. **Offline Fallback Resilience:**
   - If the backend is not running, the dashboard automatically loads verified calculation results from [`test_digital_twin.py`](file:///c:/Users/priya/OneDrive/Desktop/PCOS/src/integration/test_digital_twin.py) so the interface is never broken or blank.

3. **Multi-Curve Trajectory Visualizer:**
   - Quick presets: Gonadotropins (`LH_bld`, `FSH_bld`), Ovarian Steroids (`E2`, `P4`), Antral Follicles (`AF1`, `AF2`, `AF3`, `PrF`), Feedback Peptides (`InhB`, `InhA`).
   - Single-species inspector: Select any individual species from all 45 simulated model variables.
   - Dual viewing modes: Overlay (Baseline vs. Scenario) and Delta Only ($\Delta$).

4. **All 45 Model Species Comparison Table:**
   - Instant search filtering.
   - Category filter pills (Hormones, Follicles, Luteal, Pituitary, All).
   - Computes: Baseline Final, Scenario Final, Final Delta, % Relative Change, Maximum Absolute Difference, Mean Absolute Difference, and RMSE.

5. **Test Assertion Verification Panel:**
   - Live checklist verifying all 8 unit assertions from `test_digital_twin.py`.
