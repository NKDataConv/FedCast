# FedCast: Federated Learning for Time Series Forecasting

FedCast is a Python framework designed for time series forecasting using federated learning. It leverages the powerful [Flower (flwr)](https://flower.ai/) framework to enable privacy-preserving, decentralized model training on distributed time series data.

The core goal of FedCast is to provide a modular, extensible, and easy-to-use platform for researchers and practitioners to develop and evaluate personalized federated learning strategies for time series analysis.

## Key Features

- **Federated Time Series Forecasting**: Train models on time-series data without centralizing it.
- **Built on Flower**: Extends the robust and flexible Flower framework.
- **Modular Architecture**: Easily customize components like data loaders, models, and aggregation strategies.
- **Personalization**: Supports various strategies for building models tailored to individual clients.
- **Synthetic Data Generation**: Includes tools to simulate heterogeneous client data for robust testing.

## Getting Started

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd FedCast
    ```

2.  **Install dependencies:**
    This command will create a virtual environment and install all necessary packages, including development dependencies like `pytest`.
    ```bash
    poetry install
    ```

## Development

### Running Tests

To ensure the reliability and correctness of the framework, we use `pytest` for testing.

To run the full test suite, execute the following command from the root of the project:

```bash
poetry run pytest
```

This will automatically discover and run all tests located in the `tests/` directory.


## Run MLFlow UI:
```bash
mlflow ui --host 127.0.0.1 --port 5000
````


## Supporters

This project is supported by the German Federal Ministry of Education and Research (BMBF). We are grateful for their support, without which this project would not be possible.

<img src="logo-bmbf.svg" alt="BMBF Logo" width="250"/>
