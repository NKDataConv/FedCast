# Contributing to FedCast

Thank you for your interest in contributing to FedCast! This document provides guidelines and information for contributors to help make the contribution process smooth and effective.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Types of Contributions](#types-of-contributions)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style and Standards](#code-style-and-standards)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project follows the Apache License 2.0. By participating, you agree to uphold this code of conduct. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.9.2 to 3.13.2
- [Poetry](https://python-poetry.org/) for dependency management
- Git for version control

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/FedCast.git
   cd FedCast
   ```

2. **Install Dependencies**
   ```bash
   poetry install
   ```

3. **Activate the Virtual Environment**
   ```bash
   poetry shell
   ```

4. **Verify Installation**
   ```bash
   poetry run pytest
   ```

## Project Structure

FedCast follows a modular architecture with clear separation of concerns:

```
fedcast/
├── cast_models/           # Model implementations (MLP, Linear, etc.)
├── datasets/             # Dataset loaders for various time series data
├── experiments/          # Experiment scripts and configurations
├── federated_learning_strategies/  # FL strategies (FedAvg, FedProx, etc.)
└── telemetry/           # MLflow logging and experiment tracking

tests/                   # Test suite
├── test_dataset_*.py   # Dataset-specific tests
├── test_fed*.py        # Strategy-specific tests
└── test_datasets_general.py  # General dataset tests
```

### Key Components

- **Models** (`fedcast/cast_models/`): Neural network architectures for time series forecasting
- **Datasets** (`fedcast/datasets/`): Data loaders for various time series domains (ECG, stocks, weather, etc.)
- **Strategies** (`fedcast/federated_learning_strategies/`): Federated learning aggregation strategies
- **Experiments** (`fedcast/experiments/`): Experiment configurations and grid search scripts
- **Telemetry** (`fedcast/telemetry/`): MLflow integration for experiment tracking

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new models, datasets, or strategies
3. **Documentation**: Improve documentation, add examples, or tutorials
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize existing code
6. **Examples**: Add new experiment examples

### Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   poetry run pytest
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_dataset_stocks.py

# Run with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=fedcast
```

### Writing Tests

- Follow the existing test patterns in the `tests/` directory
- Test both success and failure cases
- Use descriptive test names
- Aim for high test coverage for new features

### Test Structure

- **Dataset Tests**: Test data loading, preprocessing, and client partitioning
- **Strategy Tests**: Test federated learning strategies and aggregation
- **Model Tests**: Test model initialization, training, and inference
- **Integration Tests**: Test end-to-end workflows

## Code Style and Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and reasonably sized

### Documentation Standards

- Use Google-style docstrings
- Include examples in docstrings for complex functions
- Document all public APIs
- Keep README files up to date

### Example Docstring Format

```python
def load_dataset(client_id: int, num_clients: int) -> Dataset:
    """Load dataset for a specific client in federated learning setup.
    
    Args:
        client_id: Unique identifier for the client (0 to num_clients-1)
        num_clients: Total number of clients in the federation
        
    Returns:
        Dataset object containing client-specific data
        
    Raises:
        ValueError: If client_id is out of valid range
        
    Example:
        >>> dataset = load_dataset(client_id=0, num_clients=10)
        >>> print(len(dataset))
        1000
    """
```

## Documentation

### Adding New Datasets

When adding a new dataset:

1. Create a new file in `fedcast/datasets/`
2. Follow the existing dataset pattern:
   - Implement `load_dataset(client_id, num_clients)` function
   - Add proper docstrings and type hints
   - Include data downloading and caching logic
   - Add client partitioning logic
3. Add corresponding tests in `tests/`
4. Update the dataset registry if applicable

### Adding New Models

When adding a new model:

1. Create a new file in `fedcast/cast_models/`
2. Inherit from appropriate base classes
3. Implement required methods (forward, get_parameters, set_parameters)
4. Add tests for the new model
5. Update the model registry

### Adding New Strategies

When adding a new federated learning strategy:

1. Create a new file in `fedcast/federated_learning_strategies/`
2. Follow Flower's strategy interface
3. Implement aggregation logic
4. Add comprehensive tests
5. Consider adding to the grid experiment

## Submitting Changes

### Pull Request Process

1. **Create a Pull Request**
   - Use a descriptive title
   - Provide a detailed description of changes
   - Reference any related issues

2. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Test improvement
   - [ ] Performance optimization

   ## Testing
   - [ ] Tests pass locally
   - [ ] New tests added for new functionality
   - [ ] All existing tests still pass

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No breaking changes (or clearly documented)
   ```

3. **Review Process**
   - Maintainers will review your PR
   - Address any feedback promptly
   - Keep PRs focused and reasonably sized

### Commit Message Format

Use clear, descriptive commit messages:

```
Add: new ECG dataset loader with MIT-BIH support
Fix: handle edge case in FedProx aggregation
Update: improve documentation for model registry
```

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Version number updated in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Documentation**: Check existing documentation and examples

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to FedCast! Your contributions help make federated learning for time series forecasting more accessible and powerful.
