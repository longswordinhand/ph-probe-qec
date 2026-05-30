# ph-probe-qec AGENTS.md

**Focus**: Quantum error correction probing
**Python env**: Uses `.venv/` (project-local virtualenv, NOT conda)

## Setup

```bash
cd ~/projects/ph-probe-qec
source .venv/bin/activate
```

## Key Directories

| Path | Purpose |
|------|---------|
| `src/` | Core modules: `decoding/`, `enumeration/`, `geometry/`, `persistence/`, `syndrome/` |
| `experiments/` | Experiment scripts |
| `tests/` | pytest tests (has `.pytest_cache/`) |
| `manuscript/` | Paper draft |
| `results/` | Experiment outputs |
| `ideas/` | Research notes |

## Important Notes

- Uses **project-local `.venv/`** — NOT a conda environment. Activate with `source .venv/bin/activate`
- Has pytest setup — run tests with `pytest tests/`
- `src/` is organized by domain: decoding, enumeration, geometry, persistence, syndrome
- Do NOT commit `results/` or `.pytest_cache/`
