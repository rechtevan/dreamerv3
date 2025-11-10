# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Repository Context

**This is a fork**: `rechtevan/dreamerv3` (forked from `danijar/dreamerv3`)

**Purpose**: This fork is maintained for:

- Code quality improvements
- Security enhancements
- Bug fixes
- Test coverage expansion
- Documentation improvements

**Upstream**: Changes will be contributed back to `danijar/dreamerv3` via pull requests.

**Issue Tracking**: Report issues to the upstream repository at
[danijar/dreamerv3](https://github.com/danijar/dreamerv3/issues).

______________________________________________________________________

## Local Development Conventions

**`.local/` Directory**: Used for AI-generated analysis, scripts, reports, and other
files that should NOT be committed to git. This directory is in `.gitignore`.

Use `.local/` for:

- Code analysis reports
- Test coverage reports and summaries (HTML, JSON, XML)
- Security scan results
- Performance benchmarking results
- Build logs for review
- Temporary scripts, diagrams, and visualizations
- Any AI-generated content not part of the codebase
- Experimental code snippets and prototypes
- Test plans and strategy documents

**Examples:**

- `.local/htmlcov/index.html` - Coverage HTML report
- `.local/coverage.json` - Coverage JSON data
- `.local/security-scan.txt` - Security findings
- `.local/performance-analysis.md` - Performance metrics
- `.local/test-plan.md` - Test strategy documents

______________________________________________________________________

## Overview

DreamerV3 is a scalable and general reinforcement learning algorithm that learns a world
model from experiences and uses it to train an actor-critic policy from imagined
trajectories. This is a JAX-based reimplementation that masters diverse control tasks
with fixed hyperparameters.

**Key Technologies:**

- **Language**: Python 3.11+
- **ML Framework**: JAX with Optax, Chex, Einops, jaxtyping, Ninjax
- **License**: MIT License
- **Lines of Code**: ~10,360 Python LOC across 58 files

______________________________________________________________________

## Installation & Setup

### Standard Installation

```bash
# Install JAX with CUDA support first
# See: https://github.com/google/jax#pip-installation-gpu-cuda
pip install jax[cuda12]==0.4.33

# Install dependencies
pip install -U -r requirements.txt

# Optional: Install in editable mode with dev dependencies
pip install -e .[dev]
```

### Docker Installation

```bash
# Build image
docker build -f Dockerfile -t dreamerv3 .

# Run with GPU support
docker run -it --rm --gpus all -v ~/logdir:/logdir dreamerv3 \
  python dreamerv3/main.py --logdir /logdir/{timestamp} --configs crafter
```

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks (recommended)
pre-commit install

# Verify installation
pytest embodied/tests/ -v
```

______________________________________________________________________

## Core Commands

### Training

Basic training command:

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs <config_name> \
  --run.train_ratio 32
```

**Common config options:**

- `--configs atari --task atari_pong` - Train on Atari tasks
- `--configs crafter` - Train on Crafter environment
- `--configs dmc_vision --task dmc_walker_walk` - Train on DeepMind Control Suite
- `--configs minecraft --task minecraft_diamond` - Train on Minecraft
- `--configs debug` - Fast debugging mode (reduced network/batch sizes, shorter
  intervals)

**Override configs** from `dreamerv3/configs.yaml`:

```bash
--batch_size 16 --jax.platform cpu
```

**Chain multiple configs** (later ones override earlier):

```bash
--configs crafter size50m
```

### Viewing Results

```bash
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Metrics are also written as JSONL files in the logdir (`metrics.jsonl`, `scores.jsonl`).

### Platform Selection

- **GPU (default)**: `--jax.platform cuda`
- **CPU**: `--jax.platform cpu`
- **TPU**: `--jax.platform tpu`

______________________________________________________________________

## Testing

### Running Tests

```bash
# Run all tests
pytest embodied/tests/ -v

# Run specific test file
pytest embodied/tests/test_train.py -v

# Run with coverage (save to .local/)
pytest --cov=dreamerv3 --cov=embodied \
  --cov-report=term \
  --cov-report=html:.local/htmlcov \
  --cov-report=json:.local/coverage.json \
  embodied/tests/

# View HTML coverage report
open .local/htmlcov/index.html  # macOS
xdg-open .local/htmlcov/index.html  # Linux
```

### Coverage Scope (What's Measured)

Coverage targets (**90% overall, 80% minimum per module/function/class**) apply to
**core production code only**:

✅ **Production code requiring coverage:**

- `dreamerv3/agent.py` - Agent training/policy logic
- `dreamerv3/rssm.py` - World model (RSSM, Encoder, Decoder)
- `embodied/core/` - Core abstractions (Driver, Replay, wrappers, etc.)
- `embodied/jax/` - JAX implementations (Agent base, nets, optimizers, heads)
- `embodied/run/` - Training loops (train.py, train_eval.py, etc.)
- `embodied/envs/` - Environment wrappers

❌ **Excluded from coverage requirements** (per `.coveragerc`):

- Visualization utilities (`plot.py`, `scores/view.py`, `*/viz/*`)
- Interactive modules (`*/interactive.py`, `*/manual_*`)
- Test files (`embodied/tests/*`, `embodied/perf/*`, `*_test.py`)
- Debug/development code (`if __name__ == "__main__"`)
- Docker entrypoint scripts

**Rationale**: Coverage focuses on core functionality used in production (agent, world
model, training loops, environments). Visualization, interactive tools, and performance
benchmarks are tested manually.

### Test Organization

```text
embodied/
├── tests/                 # Unit tests (pytest)
│   ├── test_driver.py     # Episode management, step sequencing
│   ├── test_replay.py     # Replay buffer functionality
│   ├── test_train.py      # Training loop integration tests
│   ├── test_parallel.py   # Parallel training tests
│   ├── test_sampletree.py # Data structure tests
│   ├── test_layer_scan.py # JAX layer scanning tests
│   └── utils.py           # Test utilities and helpers
│
└── perf/                  # Performance tests (benchmarking)
    ├── test_driver.py
    ├── test_replay.py
    ├── test_bandwidth.py
    └── test_distr.py
```

______________________________________________________________________

## Code Quality and Linting

### Python Code Quality

**Ruff** (linting and formatting):

```bash
# Check code (no changes)
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Check specific files
ruff check dreamerv3/agent.py embodied/core/driver.py
```

**MyPy** (static type checking):

```bash
# Type check main library code
mypy dreamerv3/ embodied/

# Type check specific module
mypy dreamerv3/agent.py

# Generate type coverage report
mypy --html-report .local/mypy-report dreamerv3/ embodied/
```

**Configuration**: `pyproject.toml` contains settings for:

- Ruff (line length: 88, select/ignore rules)
- MyPy (strict mode, exclude patterns)
- Coverage (source paths, omit patterns)

### Markdown Quality

**mdformat** (auto-formatting):

```bash
# Format markdown files
mdformat *.md docs/

# Check without modifying
mdformat --check *.md
```

**pymarkdownlnt** (linting):

```bash
# Lint markdown files
pymarkdownlnt scan *.md

# Lint with specific config
pymarkdownlnt --config .pymarkdown.toml scan *.md
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit.

```bash
# One-time setup: Install pre-commit hooks
pip install -e .[dev]
pre-commit install

# Now hooks run automatically on 'git commit'
# To skip hooks (not recommended): git commit --no-verify

# Run hooks manually on all files
pre-commit run --all-files

# Run specific hook on all files
pre-commit run ruff --all-files
pre-commit run mypy --all-files
pre-commit run mdformat --all-files

# Run hooks on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate

# Uninstall hooks (removes from .git/hooks)
pre-commit uninstall
```

**Pre-commit Hook Features:**

- **Ruff** - Lints and formats Python code, auto-fixes most issues
- **Ruff-format** - Formats Python code to consistent style
- **MyPy** - Type checks library code (excludes tests)
- **mdformat** - Formats markdown files with GFM support
- **markdownlint-cli2** - Lints markdown for style and correctness
- **Standard hooks** - Trailing whitespace, EOF fixer, YAML validation

**Configuration files:**

- `.pre-commit-config.yaml` - Hook definitions and versions
- `pyproject.toml` - Tool configurations (ruff, mypy, coverage)
- `.mdformat.toml` - Markdown formatting options
- `.markdownlint-cli2.yaml` - Markdown linting rules

______________________________________________________________________

## CI/CD and Security

### Automated Workflows

The repository uses GitHub Actions for continuous integration and security scanning:

**CodeQL Security Scanning** (GitHub Default):

- Automated security vulnerability detection for Python code
- Runs automatically on push to main
- Results viewable in GitHub Security tab → Code scanning alerts
- Uses GitHub's default security query suite
- Save results to `.local/security/` for review

**Static Analysis** (`.github/workflows/static-analysis.yml`):

- **Ruff Linting**: Checks Python code for style issues and potential bugs
- **Ruff Formatting**: Verifies code formatting consistency
- **MyPy Type Checking**: Validates type hints and catches type errors
- **Markdown Formatting**: Ensures consistent markdown file formatting
- **Markdown Linting**: Checks markdown files for syntax and style issues
- All jobs run in parallel for fast feedback
- Runs on: push to main, pull requests

**Test Coverage** (`.github/workflows/coverage.yml`):

- Runs pytest with coverage reporting
- Uploads results to Codecov
- Generates HTML/JSON reports to `.local/`
- Runs on: push to main, pull requests

**Docker Tests** (`.github/workflows/docker-tests.yml`):

- Validates tests pass in Docker environment
- Ensures reproducible test execution
- Tests GPU/CPU configurations
- Runs on: all pushes and pull requests

______________________________________________________________________

## Architecture

### High-Level Structure

The codebase is organized into two main packages:

**1. `dreamerv3/`** - The DreamerV3 agent implementation (4 files)

- `agent.py` (18KB): Main Agent class with training/policy logic, loss computation
- `rssm.py` (13.6KB): Recurrent State-Space Model (world model core), Encoder, Decoder
- `main.py` (9.4KB): Entry point, config loading, orchestration of training scripts
- `configs.yaml`: All hyperparameters and environment-specific configs

**2. `embodied/`** - Reusable RL infrastructure (51 files, framework-agnostic design)

- `core/` (11 files): Base classes (Driver, Replay, wrappers, limiters, selectors, etc.)
- `run/` (5 files): Training scripts (train.py, train_eval.py, parallel.py,
  eval_only.py)
- `envs/` (15 files): Environment wrappers for various simulators
- `jax/` (9 files): JAX-specific implementations (Agent base, nets, optimizers, heads)
- `tests/` (7 files): Unit tests covering core functionality
- `perf/` (4 files): Performance benchmarks

### Key Components

**World Model (RSSM)**:

- **Encoder**: Converts observations to categorical latent tokens
- **Dynamics (RSSM)**: Predicts future latent states given actions
  - Deterministic state (deter): GRU-like recurrent state (configurable size)
  - Stochastic state (stoch): Categorical distributions (classes × stoch dimensions)
- **Decoder**: Reconstructs observations from latent states
- **Reward head**: Predicts rewards from latent features
- **Continue head**: Predicts episode termination probabilities

**Actor-Critic**:

- **Policy (pol)**: Outputs action distributions from imagined trajectories
- **Value (val)**: Estimates state values for policy optimization
- **Slow value (slowval)**: Target network for stable training (exponential moving
  average)

**Training Flow**:

1. `embodied.Driver` collects environment experience with parallel workers
1. Experience stored in `embodied.replay.Replay` buffer with configurable selectors
1. `agent.train()` samples batches, computes losses (world model + actor-critic)
1. World model trained on reconstruction, dynamics, reward prediction
1. Actor-critic trained on imagined rollouts using the world model (λ-returns)
1. Metrics logged, checkpoints saved periodically

### Config System

All hyperparameters defined in `dreamerv3/configs.yaml`:

- `defaults`: Base configuration used by all tasks
- **Task-specific configs**: `atari`, `crafter`, `dmc_vision`, `minecraft`, `dmlab`,
  etc.
- **Model size configs**: `size1m`, `size12m`, `size25m`, `size50m`, `size100m`,
  `size200m`, `size400m`
- **Special configs**: `debug` (fast iteration), `multicpu` (CPU parallelism)

Configs are merged in order specified via `--configs` flag using regex-based overrides
for nested parameters (e.g., `.*\.rssm: {deter: 512, hidden: 256}`).

### JAX and NinjaX

- Uses **JAX** for automatic differentiation and hardware acceleration
- **ninjax** (nj) library for stateful modules with functional semantics
- Computation dtype configurable: `bfloat16` (default), `float32`, `float16`
- Supports device parallelism via `jax.policy_devices` and `jax.train_devices`
- JIT compilation enabled by default (`jax.jit: True`)

### Replay Context

When `replay_context > 0`:

- Agent state (encoder/dynamics/decoder carries) saved per timestep in replay buffer
- Allows training to resume from mid-episode states
- Reduces burn-in time for recurrent models
- Updates written to replay buffer after training steps
- Particularly useful for long sequences and recurrent state management

______________________________________________________________________

## Development Guidelines

### Adding New Environments

1. Create wrapper in `embodied/envs/` implementing `embodied.Env` interface:

   - Define `obs_space` and `act_space` properties
   - Implement `step()`, `reset()`, `close()` methods
   - Handle episode boundaries (`is_first`, `is_last`, `is_terminal`)

1. Register in `make_env()` constructor dict in `dreamerv3/main.py`:

   ```python
   ctor = {
       "atari": "embodied.envs.atari:Atari",
       "yourenv": "embodied.envs.yourenv:YourEnv",  # Add here
   }[suite]
   ```

1. Add environment-specific config block in `dreamerv3/configs.yaml`:

   ```yaml
   yourenv:
     task: yourenv_task_name
     env.yourenv: {size: [64, 64], param1: value1}
     run: {steps: 1e6, train_ratio: 256}
   ```

1. Add any dependencies to `requirements.txt` and `Dockerfile`

1. Add tests in `embodied/tests/test_yourenv.py`

### Modifying the Agent

- **Agent logic**: `dreamerv3/agent.py`

  - `policy()`: Inference-time policy selection
  - `train()`: Training step with batch processing
  - `loss()`: Loss computation (world model + actor-critic)
  - Helper functions: `imag_loss()`, `repl_loss()`, `lambda_return()`

- **Network architectures**: `dreamerv3/rssm.py` and `embodied/jax/nets.py`

  - RSSM implementation with encoder/decoder
  - Neural network building blocks (Linear, Conv2D, MLP, etc.)

- **Heads** (reward/value/policy): `embodied/jax/heads.py`

  - MLPHead for scalar/vector outputs
  - Distribution outputs (categorical, normal, etc.)

- **Loss functions**: `agent.py`

  - `agent.loss()`: Main loss aggregation
  - `imag_loss()`: Imagination-based actor-critic loss
  - `repl_loss()`: Replay-based value loss

### Checkpoint Management

- Checkpoints saved to `{logdir}/ckpt/checkpoint.pkl`
- Resume training by reusing same `--logdir` (auto-loads checkpoint)
- Load specific checkpoint: `--run.from_checkpoint <path>`
- Checkpoint includes:
  - Agent state (network parameters)
  - Replay buffer contents
  - Step counter
  - Optimizer state

**Common checkpoint errors:**

- "Too many leaves for PyTreeDef": Config mismatch with checkpoint
- Solution: Use fresh `--logdir` or ensure config matches checkpoint exactly

### Logging and Metrics

**Output formats:**

- Terminal output (default, filtered by `--logger.filter`)
- JSONL files: `{logdir}/metrics.jsonl`, `{logdir}/scores.jsonl`
- Scope summaries (default, view with `scope.viewer`)
- Optional: TensorBoard (`--logger.outputs tensorboard`)
- Optional: WandB (`--logger.outputs wandb`)

**Key metrics:**

- `episode/score`: Total episode reward
- `episode/length`: Episode length in steps
- `train/loss/*`: Individual loss components
- `fps/policy`: Policy inference speed (steps/sec)
- `fps/train`: Training throughput (steps/sec)
- `replay/*`: Replay buffer statistics
- `usage/*`: System resource usage

______________________________________________________________________

## License Requirements

### License Type: MIT License

**IMPORTANT**: All code and contributions to this fork (rechtevan/dreamerv3) must:

- Be licensed under **MIT License** (same as upstream project)
- Maintain compatibility with MIT License terms
- Respect original copyright by Danijar Hafner (2024)

**Specific requirements:**

**Runtime dependencies**:

- Must use MIT, BSD, Apache 2.0, or similar permissive licenses
- Check license compatibility before adding dependencies
- Update `requirements.txt` with version pins

**Dev dependencies**:

- Can use any OSI-approved license (pytest, ruff, mypy, etc.)
- More flexibility since not distributed with runtime

**Avoid for runtime dependencies**:

- GPL, LGPL, AGPL (copyleft licenses incompatible with MIT)
- Proprietary or restrictive licenses

### When Suggesting Dependencies

Before recommending a new dependency, verify:

1. **Check license**: Use MIT, BSD, or Apache 2.0 licensed packages
1. **Runtime vs dev**: Stricter requirements for runtime dependencies
1. **Version compatibility**: Pin versions in `requirements.txt`
1. **JAX compatibility**: Ensure compatibility with JAX ecosystem

**Example license check:**

```bash
# Using pip-licenses (install: pip install pip-licenses)
pip-licenses --from=mixed --with-urls

# Check specific package
pip show <package-name> | grep License
```

______________________________________________________________________

## Copyright Attribution Policy

### Industry Standard Approach (No Copyright Headers)

This repository follows industry standard practices for open source contributions:

✅ **MIT LICENSE file** - Covers all code with Danijar Hafner copyright ✅ **Git history**
\- Provides detailed attribution (use corporate email if applicable) ✅ **Preserves
upstream attribution** - Maintains original license and copyright ❌ **NO copyright
headers** - Not required, matches existing codebase

This is the same approach used by thousands of MIT-licensed projects and simplifies
multi-contributor collaboration.

#### General Rule: DO NOT ADD HEADERS

**Default approach for all work:**

- **Do NOT add copyright headers** to existing files when modifying them
- **Do NOT add copyright headers** to new files
- **Use git commits** with appropriate email for attribution
- The LICENSE file provides legal coverage
- Git history provides detailed attribution record
- This maintains consistency with upstream project

#### Exception: When Headers Are Required

Only add copyright headers if:

1. Legally required by your organization for substantial contributions
1. Creating entirely new subsystems (new algorithm implementations)
1. Upstream files already have headers (preserve and do not remove)

**Format (if required):**

```python
# Copyright (c) 2024 [Your Organization]
# SPDX-License-Identifier: MIT
```

______________________________________________________________________

## Contributing to Upstream

### Pull Request Guidelines

When preparing contributions for upstream (`danijar/dreamerv3`):

1. **Branch naming**: Use descriptive names

   - `fix/issue-description`
   - `feature/new-capability`
   - `test/component-coverage`

1. **Commit messages**: Clear, concise, explanatory

   - Prefix: `fix:`, `feat:`, `test:`, `docs:`, `refactor:`
   - Reference issues: `Fixes #123` or `Related to #456`

1. **Code quality**: Ensure all checks pass

   - Run `pre-commit run --all-files`
   - Run `pytest embodied/tests/` with 80%+ coverage
   - Run `ruff check .` and `ruff format .`
   - Run `mypy dreamerv3/ embodied/`

1. **Documentation**: Update relevant docs

   - Update README.md if adding features
   - Update CLAUDE.md if changing architecture
   - Add docstrings to new functions/classes

1. **Testing**: Add tests for new functionality

   - Unit tests in `embodied/tests/`
   - Integration tests if adding new components
   - Performance tests in `embodied/perf/` if relevant

1. **Keep PRs focused**: One feature/fix per PR

   - Smaller PRs are easier to review
   - Separate refactoring from feature additions
   - Split large changes into logical commits

______________________________________________________________________

## Common Issues

**"Too many leaves for PyTreeDef" error**:

- Checkpoint incompatible with current config
- Solution: Use fresh `--logdir` or ensure config matches checkpoint exactly

**CUDA errors**:

- Often caused by out-of-memory (OOM)
- Try `--batch_size 1` to diagnose
- Check JAX/CUDA version compatibility
- Verify `nvidia-cuda-nvcc-cu12<=12.2` matches your CUDA version

**Slow debugging**:

- Use `--configs debug` to reduce model size and logging intervals
- Use `--jax.platform cpu` for quick functionality tests

**Continuing training**:

- Use same `--logdir` path - checkpoint auto-loads
- Do not change config between runs

**Import errors**:

- Ensure JAX installed first: `pip install jax[cuda12]==0.4.33`
- Install requirements: `pip install -U -r requirements.txt`
- Some environments require additional packages (see Dockerfile)

**Test failures**:

- Check Python version (requires 3.11+)
- Install dev dependencies: `pip install -e .[dev]`
- Run in clean environment to avoid conflicts

______________________________________________________________________

## Quick Reference

**Repository**: `rechtevan/dreamerv3` (fork of `danijar/dreamerv3`) **License**: MIT
License **Language**: Python 3.11+ **Framework**: JAX, Optax, Ninjax **Test Framework**:
pytest **Linting**: Ruff, MyPy **Coverage Target**: 90% overall (80% minimum per
module/function/class) for core production code **Code Style**: Black-compatible (88
char line length)

**Key Files**:

- `dreamerv3/agent.py` - Main agent implementation
- `dreamerv3/rssm.py` - World model (RSSM)
- `dreamerv3/main.py` - Training entry point
- `dreamerv3/configs.yaml` - All hyperparameters
- `embodied/core/driver.py` - Environment interaction loop
- `embodied/core/replay.py` - Experience replay buffer
- `embodied/jax/` - JAX-specific neural network implementations
