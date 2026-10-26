# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README.md with installation and usage instructions
- CONTRIBUTING.md with development guidelines
- CI/CD pipeline with GitHub Actions
- Security scanning with bandit and safety
- .env.example for environment variable configuration
- Pre-commit hooks configuration
- Type hints throughout the codebase

### Performance
- Batched per-column semantic extraction into a single LLM call per dataset
  (with per-column fallback on malformed responses)
- Synthetic-data fitting runs on a 2000-row sample instead of the full frame
- `extract_types` caps the number of unique values checked per column
- Per-dataset analysis/synthesis and local agent runs execute concurrently
  (bounded to 2 concurrent jobs)
- Local agent user message only includes columns that need a transformation
- Agent session snapshots are capped to the last 3 frames

### Fixed
- Run no longer aborts with `NameError` when the local agent hits its
  recursion limit; committed steps are kept and reported
- `WillmaLLM.chat` retries transient failures (timeout/5xx/error body) with
  backoff and raises instead of returning an empty result

### Changed
- Updated setup.cfg with proper metadata and classifiers
- Improved requirements.txt with version pinning
- Enhanced .gitignore to exclude sensitive files

### Security
- Added environment variable management for API keys
- Removed hardcoded credentials from source code
- Added security scanning to CI pipeline

## [0.0.1] - 2024-01-01

### Added
- Initial release
- Core consolidation framework
- LLM-powered feature clustering (AgentCluster)
- Feature normalization with semantic alignment (AgentNorm)
- Automated transformation code generation (AgentCode)
- Support for multiple LLM backends:
  - IBM WatsonX
  - Ollama (local models)
  - RITS custom endpoints
- Dataset analyzer for semantic and structural information extraction
- Metrics for fidelity, privacy, and statistical analysis
- YAML-based experiment configuration
- Example notebooks for common workflows
- Support for DBpedia URI-based semantic type detection
- Synthetic data generation capabilities

### Dependencies
- langchain-core 0.3.74
- langchain-ibm 0.3.15
- langchain-ollama
- litellm 1.75.5.post1
- READI 0.3.3
- SPARQLWrapper 2.0.0
- tenacity 9.1.2
- pandas >= 2.0.0
- python-dotenv >= 1.0.0

[Unreleased]: https://github.com/ibm-research/privfusion/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/ibm-research/privfusion/releases/tag/v0.0.1
