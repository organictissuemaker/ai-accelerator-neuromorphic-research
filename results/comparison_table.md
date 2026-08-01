# Route A - CPU / GPU / SNN comparison (MNIST, LeNet)

| Approach   | Model              | Device               | Backend   | Accuracy   |   Latency (ms/inf) |   Throughput (inf/s) | Workload metric   |   Workload/inf |
|:-----------|:-------------------|:---------------------|:----------|:-----------|-------------------:|---------------------:|:------------------|---------------:|
| CNN        | LeNet              | CPU (Apple M-series) | cpu       | 98.73%     |             0.039  |               25,648 | MACs              |       416520   |
| CNN        | LeNet              | CPU (Colab vCPU)     | cpu       | 98.63%     |             0.2655 |                3,766 | MACs              |       416520   |
| CNN        | LeNet              | GPU (NVIDIA T4)      | cuda      | 98.63%     |             0.0127 |               78,835 | MACs              |       416520   |
| CNN        | LeNet              | GPU (Apple MPS)      | mps       | 98.73%     |             0.0211 |               47,341 | MACs              |       416520   |
| SNN        | SpikingLeNet(T=25) | CPU (Apple M-series) | cpu       | 98.49%     |             0.9027 |                1,108 | spikes            |        10546.5 |
| SNN        | SpikingLeNet(T=25) | CPU (Colab vCPU)     | cpu       | 98.50%     |             2.9772 |                  336 | spikes            |        10464.6 |
| SNN        | SpikingLeNet(T=25) | GPU (NVIDIA T4)      | cuda      | 98.50%     |             0.7511 |                1,331 | spikes            |        10464.6 |
| SNN        | SpikingLeNet(T=25) | GPU (Apple MPS)      | mps       | 98.49%     |             0.1855 |                5,392 | spikes            |        10546.5 |
