# GitHub Package Audit Report

**Package:** `E:\DFU_Cross_Domain_Research_GITHUB`
**Source of truth:** `E:\DFU_Cross_Domain_Research`
**Audit date:** 2026-11-26

## 1. Package summary

- Total files: 137
- Total size: 8.81 MB
- Git repository: initialized (see Phase 6)
- No credentials, raw datasets, checkpoints, virtual environments or caches included.

## 2. Included content

- `src/` -- model, data pipeline, training, evaluation modules.
- `scripts/` -- audit, reconciliation, preprocessing and evaluation scripts (no `__pycache__`).
- `tests/` -- unit and integration tests (no `__pycache__`).
- `configs/` -- cohort, dataset and preprocessing configuration.
- `data/manifests/` -- non-sensitive manifest README and image/mask pair audit CSV (large raw audit JSONs excluded; see section 4).
- `manuscript/` -- final IEEE PDF, Overleaf source ZIP, figures, verification and evidence maps.
- `docs/` -- project documentation.
- `README.md` -- professional research README.
- `requirements.txt`, `requirements_locked.txt`, `pyproject.toml` -- dependencies.
- `.gitignore` -- strengthened exclusion rules.

## 3. Excluded content and reasons

| Excluded | Reason |
|---|---|
| Raw dataset images and masks (`data/raw`, `data/interim`, `data/processed`) | Restricted medical data; licensing and privacy. |
| Dataset archives (`*.zip`, `*.tar`, etc.) | Restricted / large. |
| Model checkpoints (`*.pt`, `*.pth`, etc.) | Large binary weights; not appropriate for Git. |
| Virtual environments (`.venv`, `.venv_cuda`, `dfu-env`) | Machine-specific, large. |
| `__pycache__`, `.pytest_cache` | Generated cache. |
| `.kilo/`, Kaggle credential backups | Sensitive (contains `credentials.json`). |
| `reports/final_execution/kaggle_cred_backup/.kaggle_backup/credentials.json` | Secret credential -- explicitly excluded. |
| Large manifest audit JSONs (cross_dataset_duplicate_audit.json ~56 MB, etc.) | Bulky generated audit artifacts; small CSV + README retained. |
| Temporary/recovery directories (`recovered_*`, `Seed2024_Backup*`, `kaggle_recheck_*`) | Recovery artifacts; not needed for reproducible source. |
| `logs/`, `*.log` | Runtime logs. |

## 4. Security audit

- No `credentials.json`, `kaggle.json`, `access_token`, `.env` or `.kaggle_backup/` present.
- No virtual environments or cache directories present.
- No raw medical images, masks or dataset archives present.
- `.gitignore` strengthened with explicit secret, environment, cache, checkpoint, archive and raw-data exclusions.
- Machine-specific paths in scripts were not altered (source preserved); the `.gitignore` and README document configuration expectations.

## 5. Scientific integrity

- README reports only verified results from authoritative evaluation artifacts.
- A discrepancy between the task brief's seed labels (Seed 1337 = 0.2365, Seed 2024 = 0.2203) and the authoritative artifacts (Seed 1337 = 0.2203, Seed 2024 = 0.2365) is explicitly documented in the README and `manuscript/EVIDENCE_TO_MANUSCRIPT_MAP.md`; authoritative values are used.
- No claim of clinical validation, multi-centre generalization, or successful uncertainty-based failure detection is made.
- FUSeg 2021 and DFUC 2022 are stated as not experimentally used.

## 6. Git status

See `FINAL_HANDOFF_INVENTORY.md` and the terminal output captured during initialization.