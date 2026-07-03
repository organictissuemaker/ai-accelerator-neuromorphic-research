# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title: Efficient Processing of Deep Neural Networks:
A Tutorial and Survey**
- **Authors: Vivienne Sze, Senior Member, IEEE, Yu-Hsin Chen, Student Member, IEEE, Tien-Ju Yang, Student
Member, IEEE, Joel Emer, Fellow, IEEE **
- **Year: 13 Aug 2017 **
- **Venue: arXivLabs ** (e.g., ISCA, IEEE, Nature, arXiv)
- **Link: https://arxiv.org/abs/1703.09039 **

---

## What Problem Does It Solve?

*In 1–2 sentences: what gap or challenge does this paper address?*
Readers will learn following concepts in this article:
- DNNs require enormous computation and memory access, making them slow and power-hungry on conventional hardware (CPUs/GPUs). This paper surveys how to design hardware that runs DNNs efficiently — low power, low latency — without sacrificing accuracy.

---

## Key Contribution

*What did they build, prove, or show that was new?*
Provides comprehensive tutorial and survey about the recent advances towards the goal of enabling efficient processing of DNNs.  

Sections: 
- II: why DNN's are important
   - training data to choose which data to use
   - DNNs are capable of learning high-level features with more
complexity and abstraction than shallower neural networks
   - neurons (circles) & weights (connecting lines)
   - each group of connecting lines are layers
   - backpropagation: reading data patterns from right to left rather than left to right
   - uses: image and video, speech and language, medical, game play, robotics 
- III: overview of basic components of DNN's & popular DNN models 
   - two major forms of DNN networks: 
     - feed forward: data moves strictly in one direction (input layer -> hidden layers -> output layers) w/ no loops
     - recurrent: data flows forward, but hidden layers also contain feedback loops (allows network to pass past information back into itself)
- IV: resources used for DNN research and development
- V: hardware platforms used to process DNNs & optimizations used to improve throughput & energy efficiency w/o impacting application accuracry 
- VI: how mixed-signal circuits & memory tech can be used for near-data processing to address expensive data movement 
- VII: various joint algorithms & hardware optimizations performed on DNNs 
- VIII: key metrics when comparing DNN designs

---

## Architecture / Method (if applicable)

*Describe the chip, system, or algorithm design. Include a sketch or describe key components.*

- Component 1: Memory hierarchy — registers → local buffers → global buffer → DRAM. Key insight: DRAM access costs ~200x more energy than a computation, so minimizing data movement is the #1 design goal
- Component 2: Dataflow strategies — how data moves between memory and compute units. Main types: weight stationary, output stationary, row stationary (Eyeriss)
- Key design decision: Every architecture trade-off comes down to reducing how often you read/write from expensive memory

---

## Results

*What numbers matter? Fill in what they measured.*

- | Metric | Their Result | Compared To |
- | Energy per DRAM access | ~200 pJ | ~1 pJ for register access |
- | Eyeriss power | 278 mW running AlexNet | 10x better than mobile GPU |
- | Eyeriss throughput | 35 fps | Mobile GPU baseline |

---

## How This Connects to My Project

This paper explains why AI accelerators exist and what problem they solve — data movement cost. My Python simulations will measure power and latency across architectures, and this paper gives me the vocabulary and framework to interpret those results. The dataflow concepts directly inform how I structure my architecture comparison.

---

## What I'd Explore Further

How does the row-stationary dataflow from Eyeriss compare to how neuromorphic chips like Loihi handle data movement? 
    - 


SNNs fire sparsely — does event-driven computation naturally solve the data movement problem this paper describes?
    - 
---

## 2-Sentence Explanation

I studied Sze et al. 2017, which is the foundational survey on efficient DNN hardware — it taught me that the core bottleneck in AI chips is data movement, not computation, which costs 200x more energy per DRAM access than a register access. That insight directly shaped how I designed my architecture comparison simulator, where I measure power trade-offs across CPU, GPU, and neuromorphic designs.



