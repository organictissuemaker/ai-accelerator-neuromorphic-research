# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title: Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks **
- **Authors: Yu-Hsin Chen, Tushar Krishna, Joel Emer, Vivienne Sze **
- **Year: 2016 **
- **Venue:ISCA (International Symposium on Computer Architecture) 
- **Link: https://dl.acm.org/doi/10.1145/3007787.3001177 **

---

## What Problem Does It Solve?

Finding a dataﬂow that supports parallel processing withminimal data movement cost is crucial to achieving energy-efﬁcient CNN processing without compromising accuracy. 

Prior dataflows (weight stationary, output stationary) optimized for only one type of data reuse at a time — Eyeriss's row-stationary approach minimizes movement across all data types simultaneously.
---

## Key Contribution

This paper introduces row stationary (RS), which minimizes data movement energy consumption on a spatial architecture. 

RS dataﬂowcan adapt to different CNN shape conﬁgurations and reducesall types of data movement through maximally utilizing the processing engine (PE) local storage, direct inter-PE communi-cation and spatial parallelism. 

RS maximizes reuse at three levels:
 - (1) PE local scratchpad 
 - (2) direct PE-to-PE communication (no going back to global buffer)
 - (3) spatial parallelism across the 168 PE array.

---

## Architecture / Method (if applicable)

*Describe the chip, system, or algorithm design. Include a sketch or describe key components.*

- Component 1: 168 PE spatial array — processing elements arranged in a grid, each with its own local scratchpad memory. PEs communicate directly with neighbors without going back to global memory, enabling inter-PE data reuse
- Component 2: Reconfigurable multicast on-chip network — routes data from the global buffer to PEs efficiently. One weight value can be broadcast to multiple PEs simultaneously, drastically reducing how many times data is fetched from memory
- Key design decision: Row-stationary dataflow — maps one row of a CNN filter computation onto one PE. Each PE accumulates partial sums locally, reuses weights and input activations from its scratchpad, and passes data to neighboring PEs directly. This minimizes movement at all three levels simultaneously: local scratchpad, inter-PE, and global buffer

---

## Results

*What numbers matter? Fill in what they measured.*

 Metric: Their Result vs. Compared To 

Power:278 mW vs. ~2,780 mW mobile GPU (10x better)
Throughput: 35 fps running AlexNet vs. Mobile GPU baseline
Data movement: energyMinimized across all types vs. Prior dataflows optimized only one type at a time

---

## How This Connects to My Project

Eyeriss is the concrete hardware example behind everything Sze et al. described theoretically. The 278 mW and 35 fps numbers are real data points for my arch_comparison.py power-latency plot. The row-stationary dataflow shows me how hardware structure decisions (PE layout, local memory, on-chip network) directly determine energy efficiency — which is the design trade-off my simulations are evaluating. When I compare neuromorphic vs accelerator architectures, Eyeriss represents the best-in-class traditional accelerator on that curve. 

---

## What I'd Explore Further

Eyeriss v2 (2019) introduced a more flexible dataflow for irregular/sparse networks like MobileNet. How does row-stationary perform on sparse SNNs compared to Loihi's event-driven approach — and is there a hybrid that gets the best of both?

---

## 2-Sentence Explanation

I studied Eyeriss, MIT's AI accelerator chip, which achieves 10x better energy efficiency than a mobile GPU by using a row-stationary dataflow that reuses data at three levels simultaneously — local PE memory, direct PE-to-PE communication, and spatial parallelism across 168 processing elements. This directly informed my architecture comparison simulator, where Eyeriss's 278 mW / 35 fps benchmark is one of the design points I plot against CPU, GPU, and neuromorphic architectures on a power-latency trade-off curve

