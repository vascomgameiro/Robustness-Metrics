# Robustness Metrics Benchmarking

## Overview
This repository contains the code and final report for the project *Benchmarking Robustness Measures*, which evaluates various robustness metrics in deep learning models. The primary focus of the project is to explore the relationship between model generalization, uncertainty, and robustness metrics under different out-of-distribution scenarios.

## Project Structure
```
Robustness-Metrics/
├── data/                     # Contains processed data and results
│   └── results/              # Metrics calculated for different models
│       ├── norms.csv         # Norm metrics
│       ├── ood_performance.csv # Out-of-distribution performance metrics
│       ├── performance_gap.csv # Performance gap metrics
│       ├── proportional_performance.csv # Proportional performance metrics
│       ├── sharpness.csv     # Sharpness metrics
│       └── train_val_gap.csv # Training and validation gap metrics
├── notebooks/                # Jupyter notebooks for analysis and visualization
│   └── results_analysis.ipynb # Analysis of robustness metrics
├── references/               # Reference materials and related literature
├── reports/                  # Final report
│   ├── figures/              # Figures used in the report
│   └── AECD_Benchmark_Robustness_Metrics_Report.pdf
│   └── AECD_Benchmark_Robustness_Metrics_Presentation.pdf
├── src/                      # Source code for training, evaluation, and metric computation
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── process_data.py       # Data processing scripts
│   ├── make_cifar_c.py       # CIFAR-C generation script
│   ├── model_constructor.py  # Model architectures and definitions
│   ├── pytorch_trainer.py    # Training pipeline
│   ├── attack_acc.py         # Implementation of adversarial attacks
│   ├── measures_norm.py      # Norm metrics computation
│   ├── measures_performance.py # Performance metrics computation
│   ├── measures_sharpness.py # Sharpness metrics computation
│   ├── load_results.py       # Results loading scripts
│   ├── visualization.py      # Visualization utilities
│   └── utils.py              # General utilities
├── .gitignore                # Git ignore file
├── LICENCE                   # License information
├── README.md                 # Project overview and instructions
└── requirements.txt          # Required dependencies
```

## Highlights
- **Reports:** The final report summarizing the findings is located in the `reports/` folder.
- **Data:** The `data/` folder contains processed data and results, including metrics for different models.
- **Results:** The `results/` subfolder provides detailed metrics such as norms, sharpness, performance gaps, and out-of-distribution performance.

## Metrics and Evaluation
We calculated and analyzed multiple metrics to benchmark model robustness:

1. **Norm Metrics:** Measures related to model weights and their interactions.
2. **Performance Metrics:** Standard measures like accuracy and F1-score.
3. **Sharpness Metrics:** Metrics evaluating the sensitivity of the loss landscape.
4. **Out-of-Distribution Performance:** Metrics comparing in-distribution and out-of-distribution results.

The results from these metrics are summarized in the `results/` folder as CSV files, facilitating further analysis and visualization.

## References
For more details on the methodology and findings, refer to the final report located in the `reports/` folder.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
