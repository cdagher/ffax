# JAX Forward-Forward Implementation

This repository contains an implementation of the Forward-Forward (FF) algorithm using JAX. The FF algorithm is an alternative to backpropagation for training neural networks. The repository includes code for training and testing on the MNIST dataset, along with visualizations and sample data.

## Repository Structure

- **`ffax/`**: Includes additional utilities and modules for working with the FF algorithm.
  - `nn/`: Contains neural network modules such as `module.py` and `network.py`.
  - `quickstart.ipynb`: A Jupyter Notebook for quickly getting started with the FF algorithm.

## Getting Started

### Prerequisites

Ensure you have Python installed along with the required dependencies. Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Acknowledgements

- The MNIST dataset is used for training and testing the FF algorithm.
- The FF algorithm is based on the paper "Forward-Forward: Some Preliminary Results" by Geoffrey Hinton.

## Citation

If you find this repository useful in your research, please cite it as follows:

```bibtex
@software{ffax2025github,
    author = {Christopher Dagher},
    title = {FFAX: Forward-Forward with JAX},
    url = {https://github.com/cdagher/ffax},
    version = {0.1.0},
    year = {2025},
}
```
