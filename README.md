# PDP project : Accelerating Generative Inference with multiple GPUs

## Content
- [Installation](#installation)

- [Performance Results](#performance-results)


## Installation
Requirements:  
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

### Method : With pip
```
pip install flexgen
```

## Performance Results
You need at least four GPUs to check results.

#### Baseline (FlexGen) (OPT-1.3B, OPT-6.7B)
```
bash scripts/run_bash.sh
```

#### Ours (DP) (OPT-1.3B, OPT-6.7B)
```
bash scripts/run_ours_dp.sh
```

#### Ours (Hybrid) (OPT-1.3B, OPT-6.7B)

```
bash scripts/run_ours_hybrid.sh
```

| System | OPT-1.3B | OPT-6.7B |
| ------ | -------- | ------- |
| Baseline (FlexGen )   | 1358.41 | 546.58 | 
| Ours (DP) |  2434.33 | 532.36 | 
| Ours (Hybird)    | 2005.88 | 685.13 | 

- Hardware: four NVIDIA A6000 instance on GCP with 512GB of DRAM 
- Workload: input sequence length = 512, output sequence length = 32. The batch size is tuned to **a large value** that maximizes the generation throughput for each system.
- Metric: generation throughput (token/s) = number of the generated tokens / (time for processing prompts + time for generation).  
