# HASA: Simulation & Evaluation Framework

This repository contains the official Python simulation and visualization script for the research paper: **"HASA: A Hierarchical Aggregation and Selective-Update Algorithm for Accelerating LLM Attention on Processor-in-Memory Systems"**.

The code is designed to reproduce the key figures presented in the paper, demonstrating the performance advantages of the HASA algorithm over conventional GPU-based and naive PIM-based approaches.

**[Link to Paper PDF or ArXiv]** (<- *请在此处替换为您的论文链接*)

## Repository Structure

```
.
├── final_experiments.py   # The main simulation and plotting script
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Getting Started

Follow these instructions to set up the environment and run the simulations on your local machine.

### Prerequisites

*   Python 3.8 or newer
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUserName/YourRepoName.git
    cd YourRepoName
    ```
    (*请将 `YourUserName/YourRepoName` 替换为您的实际 GitHub 用户名和仓库名*)

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python libraries using `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run all experiments and generate all figures from the paper, simply execute the main script:

```bash
python final_experiments.py
```

The script will print the status of each experiment to the console as it runs.

## Expected Output

The script will run through 5 experiments, corresponding to Figures 3 through 7 in the paper. Upon successful completion, you will find the following high-quality PDF files in the root directory of the repository, ready for inclusion in a LaTeX document:

*   `figure_3_communication_overhead.pdf`
*   `figure_4_end_to_end_latency.pdf`
*   `figure_5_scalability.pdf`
*   `figure_6_top_k_impact.pdf`
*   `figure_7_latency_breakdown.pdf`

## Code Overview

The script `run_final_experiments.py` is self-contained and organized into three main sections:

1.  **Global Parameters:** Defines the core parameters of the simulated LLM and PIM systems.
2.  **Simulation Functions:**
    *   `simulate_data_transfer()`: Models the communication overhead.
    *   `simulate_latency()`: Models the end-to-end latency for the three system configurations (Baseline GPU, PIM-Naive, PIM-HASA). The parameters within this function have been carefully tuned to reflect the performance characteristics discussed in the paper.
    *   `simulate_latency_breakdown()`: Models the five phases of the HASA algorithm.
    *   `simulate_perplexity()`: Models the impact of the Top-K approximation on model accuracy.
3.  **Plotting Functions:**
    *   Each `plot_*()` function is responsible for running a specific experiment and generating the corresponding figure using Matplotlib and Seaborn.

> **Note:** Please be aware that this script implements a high-level performance model designed to accurately capture the performance trends, scalability, and relative differences between the architectures. It is not a cycle-approximate simulator but serves to validate the conceptual and architectural advantages of the HASA algorithm.


## License

This project is licensed under the MIT License. 
