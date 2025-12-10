# Contributing to ABC Inflation Forecasting

Thank you for your interest in contributing to this research project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ABC_Inflation_forecasting.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Format code with Black: `black src/ tests/`
- Check linting: `flake8 src/ tests/`

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Run tests: `pytest tests/ -v --cov=src`

## Documentation

- Update README.md if adding new features
- Document functions with numpy-style docstrings
- Update paper sections if relevant to research

## Research Contributions

If you're contributing research ideas:

- Describe methodology clearly
- Include references to relevant papers
- Provide empirical validation where possible
- Update bibliography (references.bib)

## Questions?

Open an issue for questions or discussions about the project.
