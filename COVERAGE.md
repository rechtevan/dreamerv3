# Test Coverage Status

**Last Updated**: 2025-11-10 **Overall Coverage**: **26.06%** (1,212 / 4,650 lines
covered)

______________________________________________________________________

## Quick Summary

- ✅ **141 tests passing**
- ❌ **55 tests failing** (API mismatches, missing dependencies)
- ⚠️ **2 test files skipped** (missing 'zerofun' module)
- **196 total tests**

See `.local/coverage-baseline.md` for full analysis (generated locally, not in git).

______________________________________________________________________

## Coverage by Priority

### 🔴 CRITICAL - Core Algorithm (0% coverage)

| Module               | Coverage  | Lines | Status      |
| -------------------- | --------- | ----- | ----------- |
| `dreamerv3/agent.py` | **0.00%** | 313   | ❌ No tests |
| `dreamerv3/main.py`  | **0.00%** | 139   | ❌ No tests |
| `dreamerv3/rssm.py`  | **0.00%** | 288   | ❌ No tests |

**Action Required**: Add tests for core DreamerV3 algorithm (Issues #8, #9)

### 🟡 MEDIUM - JAX Modules (11-54% coverage)

| Module                      | Coverage | Status |
| --------------------------- | -------- | ------ |
| `embodied/jax/transform.py` | 11.29%   | ⚠️ Low |
| `embodied/jax/agent.py`     | 15.17%   | ⚠️ Low |
| `embodied/jax/nets.py`      | 27.67%   | ⚠️ Low |
| `embodied/jax/utils.py`     | 53.90%   | ✅ OK  |

**Action Required**: Expand JAX module tests (Issue #10)

### ✅ GOOD - Core Infrastructure (54-100% coverage)

| Module                       | Coverage | Status       |
| ---------------------------- | -------- | ------------ |
| `embodied/core/selectors.py` | 54.35%   | ✅ OK        |
| `embodied/core/driver.py`    | 61.74%   | ✅ Good      |
| `embodied/core/replay.py`    | 68.98%   | ✅ Good      |
| `embodied/core/chunk.py`     | 83.33%   | ✅ Excellent |
| `embodied/core/base.py`      | 89.29%   | ✅ Excellent |

**Status**: Well-tested, maintain current coverage

______________________________________________________________________

## Known Issues

1. **Missing 'zerofun' dependency**: Blocks `test_parallel.py` and `test_train.py`
1. **API mismatches**: Many replay tests expect `.dataset()` method (deprecated)
1. **Driver test failures**: Missing 'reset' key in test expectations
1. **Replay edge cases**: Single-item buffer assertions fail

______________________________________________________________________

## Coverage Goals

**Project Goal**: 90% overall coverage with 80% minimum per module/function/class

| Phase          | Target   | Current | Gap      |
| -------------- | -------- | ------- | -------- |
| Core Algorithm | 90%+     | 0%      | +90%     |
| JAX Modules    | 90%+     | 26%     | +64%     |
| Training Loops | 90%+     | 6%      | +84%     |
| Core Infra     | 90%+     | 58%     | +32%     |
| **Overall**    | **90%+** | **26%** | **+64%** |

**Minimum Acceptable**: 80% coverage per module/function/class

______________________________________________________________________

## Running Coverage Locally

```bash
# Run tests with coverage
pytest embodied/tests/ \
  --ignore=embodied/tests/test_parallel.py \
  --ignore=embodied/tests/test_train.py \
  --cov=dreamerv3 \
  --cov=embodied \
  --cov-report=html:.local/htmlcov \
  --cov-report=term

# View HTML report
open .local/htmlcov/index.html  # macOS
xdg-open .local/htmlcov/index.html  # Linux
```

______________________________________________________________________

## Next Steps

1. ✅ **Issue #1**: Development environment setup (DONE)
1. ✅ **Issue #2**: Coverage baseline established (DONE)
1. ⏳ **Issue #3**: Implement pre-commit hooks
1. ⏳ **Issue #4-7**: Setup CI/CD workflows
1. ⏳ **Issue #8-10**: Expand test coverage for core modules

**See**: `.local/coverage-baseline.md` for detailed analysis and improvement roadmap.
