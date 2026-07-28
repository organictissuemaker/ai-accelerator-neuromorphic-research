# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title:** Networks of Spiking Neurons: The Third Generation of Neural Network Models
- **Authors:** Wolfgang Maass
- **Year:** 1997
- **Venue:** Neural Networks, vol. 10, no. 9, pp. 1659–1671
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0893608097000117

---

## What Problem Does It Solve?

It answers a foundational question: are spiking neurons — which compute with the timing of individual spikes — just a biological curiosity, or a genuinely more powerful class of computing element? Maass gives a formal theory placing them in a hierarchy and proves they raise computational power per neuron.

---

## Key Contribution

Defines the three generations of neural network models and proves spiking neurons (with temporal coding) sit at the top: they can simulate the earlier two generations efficiently, but not cheaply vice versa. Establishes that spike timing is a computational medium, not just noise. 

---

## Architecture / Method (if applicable)

- Component 1 - 1st gen (McCulloch–Pitts / threshold gates): binary in → binary out. Perceptrons, Hopfield nets. Compute Boolean functions.
- Component 2 - 2nd gen (continuous activation): sigmoid/RBF applied to a weighted sum → analog out; trainable by backprop; universal analog approximators; outputs often read as firing rates (rate coding).
- Component 3 - 3rd gen (spiking neurons): output is a spike train; computation uses the precise timing of spikes (temporal coding, e.g. time-to-first-spike), matching how biology encodes information. 
- Key design decision (of the theory): information rides on when a neuron fires. A weighted sum can be encoded in relative spike delays, so a single spiking neuron performs analog computation via timing — the mechanism behind the power results.

---

## Results


| Metric | Their Result | Compared To |
|---|---|---|
| Power(expressive) | Any 2nd-gen net with s gates → computable by O(s) spiking neurons; converse not cheap | 1st- & 2nd-gen models |
| Efficiency (neuron count) |  Certain functions (coincidence-detection / element-distinctness type) need far fewer spiking neurons | equivalent 1st/2nd-gen networks |
| Robustness | Power results hold under bounded noise in spike timing | idealized noise-free case |

---

## How This Connects to My Project

- Supplies the "why SNNs are 3rd generation" for my Task-3 synthesis — the citation behind treating event-driven computation as a property of the model, not a low-power hack.
- Thread A (AI-vs-CPU): temporal coding is what a clocked CPU cannot exploit and an event-driven accelerator (Loihi) can — the theoretical basis for my comparison.
- Algorithm → hardware requirement: if computation lives in spike timing, hardware must preserve timing (async event routing, fine temporal resolution, low jitter). The algorithm dictates a hardware capability — the essence of algorithm–hardware interaction.

---

## 2-Sentence Explanation

Neural network models come in three generations, and the third — spiking neurons that compute using the exact timing of their spikes — is provably at least as powerful as the earlier rate-based and threshold-based models, often needing far fewer neurons. That timing-based computation is the theoretical reason event-driven hardware is sparse and efficient, and it sets the idealized ceiling that my RRAM-variability experiment tests against. 
