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
    - datasets — what trains DNNs. standard benchmark: ImageNet (1.2M images, 1000 categories); the quality and size of training data directly determines what the hardware needs to handle
    - frameworks — TensorFlow, PyTorch, and Caffe are the software layers that translate a DNN model into actual hardware operations. they sit between the algorithm and the chip
    - model zoos — pre-trained models (AlexNet, ResNet, VGG) that serve as standard benchmarks for comparing hardware performance across different chips
- V: hardware platforms used to process DNNs & optimizations used to improve throughput & energy efficiency w/o impacting application accuracry 
    - four hardware tiers — CPU (flexible, slow for AI), GPU (parallel, high power), FPGA (reconfigurable, medium efficiency), ASIC (fixed-function, most efficient) — each trades flexibility for efficiency
    - dataflow strategies — weight stationary, output stationary, and row stationary determine what data stays close to compute. This is the primary lever for reducing energy without touching accuracy
    - parallelism — three types (choosing the right one matters for throughput): 
      - data parallelism (same model, different data)
      -  model parallelism (split model across chips)
      - pipeline parallelism (different layers on different hardware)
- VI: how mixed-signal circuits & memory tech can be used for near-data processing to address expensive data movement 
    - processing-in-memory (PIM) — instead of moving data to the processor, do the math inside the memory itself. almost entirely eliminates the 200 pJ data movement cost 
    - analog computation — using analog circuits (voltage, current) to perform multiply-accumulate directly, rather than converting everything to digital first. dramatically lower power but introduces noise and precision challenges
    - emerging memory technologies — SRAM, eDRAM, RRAM (memristors) each offer different trade-offs between speed, density, and energy. RRAM can both store and compute, making it a natural fit for in-memory computing
- VII: various joint algorithms & hardware optimizations performed on DNNs 
    - weight pruning: removing connections in a neural network that contribute little to accuracy (setting weights to zero). sparse networks need less computation and less data movement, directly reducing power
    - quantization: reducing numerical precision from 32-bit floating point to 8-bit or even 1-bit integers. fewer bits = smaller memory footprint, cheaper data movement, faster computation with minimal accuracy loss
    - network architecture design: designing models (MobileNet, SqueezeNet) specifically to be hardware-friendly from the start, rather than taking an existing model and trying to compress it afterward
- VIII: key metrics when comparing DNN designs
    - energy efficiency (GOPS/W) — giga-operations per second per watt. how much useful computation you get per unit of energy — the primary metric for edge devices
    - throughput vs latency — throughput (frames per second, GOPS) measures how much you can process; latency measures how fast a single sample gets an answer.
    - accuracy — top-1 and top-5 error rates on standard benchmarks (ImageNet). all three metrics must be reported together

---

## Architecture / Method (if applicable)

*Describe the chip, system, or algorithm design. Include a sketch or describe key components.*

- Component 1: Memory hierarchy — registers → local buffers → global buffer → DRAM. Key insight: DRAM access costs ~200x more energy than a computation, so minimizing data movement is the #1 design goal
- Component 2: Dataflow strategies — how data moves between memory and compute units. Main types: weight stationary, output stationary, row stationary (Eyeriss)
- Key design decision: Every architecture trade-off comes down to reducing how often you read/write from expensive memory

---

## Results

*What numbers matter? Fill in what they measured.*

| Metric | Their Result | Compared To |
|---|---|---|
| Energy per DRAM access | ~200 pJ | ~1 pJ for register access (200x more energy)|
| Eyeriss power | 278 mW running AlexNet | 10x better than mobile GPU |
| Eyeriss throughput | 35 fps | Mobile GPU baseline |

---

## How This Connects to My Project

This paper explains why AI accelerators exist and what problem they solve — data movement cost. My Python simulations will measure power and latency across architectures, and this paper gives me the vocabulary and framework to interpret those results. The dataflow concepts directly inform how I structure my architecture comparison.

---

## What I'd Explore Further
- How does the row-stationary dataflow from Eyeriss compare to how neuromorphic chips like Loihi handle data movement? 
    - Eyeriss: minimizes data movement by being smart about scheduling — it decides in advance what data to keep close and reuses it as many times as possible before fetching new data. It still runs on a clock, meaning every cycle costs energy whether or not useful work happens.
    - Loihi: doesn't schedule data movement at all. Nothing moves unless a neuron fires. The hardware is silent by default and only activates when a spike arrives. So instead of minimizing data movement (eliminates most of it)
    - Both solve the same problem but from opposite directions: Eyeriss optimizes the movement that has to happen, Loihi avoids the movement altogether.
- SNNs fire sparsely — does event-driven computation naturally solve the data movement problem this paper describes?
    - Yes in an SNN, a neuron only fires maybe 1-5% of the time. That means 95-99% of the time, there's no output, no data to move, no memory access needed.
    - The event-driven architecture physically enforces this — circuits only switch when a spike occurs, so zero spikes = zero switching = zero energy. This is why Loihi achieves 200x lower energy than a GPU on sparse tasks like keyword spotting.
    - However, for dense tasks where neurons fire frequently, the sparsity advantage disappears and neuromorphic chips lose their edge over conventional accelerators.

## 2-Sentence Explanation

I studied Sze et al. 2017, which is the foundational survey on efficient DNN hardware — it taught me that the core bottleneck in AI chips is data movement, not computation, which costs 200x more energy per DRAM access than a register access. That insight directly shaped how I designed my architecture comparison simulator, where I plan to measure power trade-offs across CPU, GPU, and neuromorphic designs.



