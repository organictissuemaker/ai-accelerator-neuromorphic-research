# AI Accelerator & Neuromorphic Computing Research

**Summer 2026 Research | SUTD Singapore | UC Berkeley EECS**

Exploring innovative design methodologies for AI accelerators and neuromorphic computing systems — integrating energy-efficient, scalable, and brain-inspired architectures for next-generation intelligent devices.

---

## Project Overview

Conventional processors (CPUs and GPUs) face fundamental limits in power efficiency and scalability for AI workloads. This research investigates two complementary approaches:

- **AI Accelerator Architecture** — energy-efficient hardware designs optimized for ML/DL inference
- **Neuromorphic Computing** — brain-inspired, event-driven architectures using Spiking Neural Networks (SNNs)

The goal is to explore hybrid architectures capable of real-time adaptive learning at the edge, relevant to robotics, autonomous vehicles, and intelligent sensing.

---

## What's in This Repo

| Folder | Contents |
|---|---|
| `literature/notes/` | Paper summaries — one `.md` file per paper |
| `simulations/snn/` | Python SNN simulator (Leaky Integrate-and-Fire neuron model) |
| `simulations/accelerator/` | Architecture comparison scripts (CPU vs GPU vs neuromorphic) |
| `results/figures/` | Power-latency trade-off plots, Weibull plots, performance charts |
| `report/` | Final literature review and technical write-up |

---

## Key Results

*To be updated as research progresses.*

---

## Tools & Libraries

- Python 3.10+
- `numpy`, `matplotlib`, `seaborn` — simulation and visualization
- `norse` or `spikingjelly` — SNN frameworks
- `scipy` — statistical modeling

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-accelerator-neuromorphic-research.git
cd ai-accelerator-neuromorphic-research

# Install dependencies
pip install -r requirements.txt

# Run the SNN simulator
python simulations/snn/lif_neuron.py

# Run architecture comparison
python simulations/accelerator/arch_comparison.py
```

---

## About

**Researcher:** Zena Wu | UC Berkeley EECS  
**Supervisor:** Dr. Shubhakar Kalya | SUTD  
**Topics:** AI Accelerators, Neuromorphic Computing, Spiking Neural Networks, Memristors, RRAM, Hardware-Software Co-Design
