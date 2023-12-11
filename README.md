# PDP project : Accelerating Generative Inference with multiple GPUs

## Content
- [Installation](#installation)
- [Usage and Examples](#usage-and-examples)
  - [Get Started with a Single GPU](#get-started-with-a-single-gpu)
  - [Run HELM Benchmark with FlexGen](#run-helm-benchmark-with-flexgen)
  - [Run Data Wrangling Tasks with FlexGen](#run-data-wrangling-tasks-with-flexgen)
  - [Scaling to Distributed GPUs](#scaling-to-distributed-gpus)
  - [API Example](#api-example)
  - [Frequently Asked Questions](#frequently-asked-questions)
- [Performance Results](#performance-results)
- [How It Works](#how-it-works)
- [Roadmap](#roadmap)

## Installation
Requirements:  
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

### Method : With pip
```
pip install flexgen
```

## Usage and Examples
You need at least four GPU to check results.

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
