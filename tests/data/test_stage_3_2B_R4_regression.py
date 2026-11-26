"""
Regression tests for Stage 3.2B-R4 Canonical Annotation Role Enforcement.
Validates:
1. Native target has exactly 2,187 primary references.
2. Native target has exactly 177 alternates.
3. No unique image hash has more than one primary reference across primary benchmarks.
4. Lowest alphanumeric Pair ID is selected for every unique image group.
5. Source training has 945 primary references and 17 alternates across 945 unique images.
6. Exact duplicate pair 0092/0335 does not increase sampling probability.
7. Internal validation has 191 eligible primary binary samples.
8. Multiclass record 0233 is excluded from primary validation.
9. Mismatched record 1484 remains diagnostic-only.
10. Native record 60841288 remains eligible.
11. Healthy controls remain outside segmentation-loss training.
12. Primary metrics count unique images, not annotation records.
13. The loader reads corrected v2 roles rather than stale v1 flags.
14. Original v1 manifest and metadata remain unchanged.
"""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
import pytest

from src.data.stage_4_dataloaders import build_stage_4_dataloaders


PROJECT_ROOT = Path("E:/DFU_Cross_Domain_Research")
V1_MANIFEST_PATH = PROJECT_ROOT / "data/processed/stage_3_2B_manifest_repair_v1/canonical_manifest_corrected.csv"
V2_MANIFEST_PATH = PROJECT_ROOT / "data/processed/stage_3_2B_manifest_repair_v2/canonical_manifest_corrected_v2.csv"
ORIG_MANIFEST_PATH = PROJECT_ROOT / "data/processed/stage_3_2B_canonical/manifests/canonical_manifest.csv"

# Known cryptographic hashes
EXPECTED_ORIG_MANIFEST_SHA256 = "a0272e7a7c1de2b9285e2f19f65172c1ade8b227c3be2165a28b09730683e314"
EXPECTED_V1_MANIFEST_SHA256 = "36d556c7ce670a88bab2b3bee98e1f5287ea5cc8539ac9219373c7197f2e52b3"


def compute_file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def v2_rows():
    with open(V2_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_01_native_target_primary_reference_count(v2_rows):
    """Test 1: Native target has exactly 2,187 primary references."""
    native_primaries = [r for r in v2_rows if r["cohort"] == "native_target" and r["is_primary_reference"] == "True"]
    assert len(native_primaries) == 2187


def test_02_native_target_alternate_count(v2_rows):
    """Test 2: Native target has exactly 177 alternates."""
    native_alts = [r for r in v2_rows if r["cohort"] == "native_target" and r["is_primary_reference"] != "True"]
    assert len(native_alts) == 177


def test_03_no_unique_image_has_multiple_primaries(v2_rows):
    """Test 3: No unique image hash has more than one primary reference in any benchmark cohort."""
    for cohort in ("source_training", "internal_validation", "native_target"):
        cohort_recs = [r for r in v2_rows if r["cohort"] == cohort]
        groups = defaultdict(list)
        for r in cohort_recs:
            groups[r["orig_img_sha256"]].append(r)
        for img_sha, recs in groups.items():
            primaries = [r for r in recs if r["is_primary_reference"] == "True"]
            assert len(primaries) == 1, f"Cohort {cohort} image {img_sha} has {len(primaries)} primaries!"


def test_04_lowest_alphanumeric_pair_id_selected(v2_rows):
    """Test 4: Lowest alphanumeric Pair ID is selected for every unique image group."""
    for cohort in ("source_training", "internal_validation", "native_target"):
        cohort_recs = [r for r in v2_rows if r["cohort"] == cohort]
        groups = defaultdict(list)
        for r in cohort_recs:
            groups[r["orig_img_sha256"]].append(r)
        for img_sha, recs in groups.items():
            lowest_pid = sorted(recs, key=lambda x: str(x["pair_id"]))[0]["pair_id"]
            primary_pid = [r["pair_id"] for r in recs if r["is_primary_reference"] == "True"][0]
            assert primary_pid == lowest_pid, f"Cohort {cohort} primary {primary_pid} != lowest {lowest_pid}"


def test_05_source_training_primaries_and_alternates(v2_rows):
    """Test 5: Source training has 945 primary references and 17 alternates."""
    train_recs = [r for r in v2_rows if r["cohort"] == "source_training"]
    assert len(train_recs) == 962
    primaries = [r for r in train_recs if r["is_primary_reference"] == "True"]
    alternates = [r for r in train_recs if r["is_primary_reference"] != "True"]
    assert len(primaries) == 945
    assert len(alternates) == 17
    unique_imgs = len(set(r["orig_img_sha256"] for r in train_recs))
    assert unique_imgs == 945


def test_06_alternate_annotation_is_not_a_training_unit():
    """Test 6: Duplicate pair 0092/0335 creates no second training sampling unit."""
    loaders = build_stage_4_dataloaders()
    train_sampler = loaders["train_sampler"]
    train_ds = loaders["train_dataset"]

    pair_ids = {r["pair_id"] for r in train_ds.records}
    assert "0092" in pair_ids
    assert "0335" not in pair_ids  # alternate excluded from training units

    # One primary per unique image hash, and one sampler index per unit.
    assert len(train_ds) == 945
    assert len({r["orig_img_sha256"] for r in train_ds.records}) == 945
    epoch_indices = list(train_sampler)
    assert len(epoch_indices) == 945
    assert len(set(epoch_indices)) == 945


def test_07_internal_validation_eligible_count():
    """Test 7: Internal validation has 191 eligible primary binary samples."""
    loaders = build_stage_4_dataloaders()
    val_ds = loaders["val_dataset"]
    assert len(val_ds) == 191


def test_08_multiclass_record_0233_excluded():
    """Test 8: Multiclass record 0233 is excluded from primary validation loader."""
    loaders = build_stage_4_dataloaders()
    val_ds = loaders["val_dataset"]
    val_pair_ids = [r["pair_id"] for r in val_ds.records]
    assert "0233" not in val_pair_ids


def test_09_mismatched_record_1484_diagnostic_only(v2_rows):
    """Test 9: Mismatched record 1484 remains diagnostic-only."""
    rec_1484 = [r for r in v2_rows if r["cohort"] == "mismatched_target" and r["pair_id"] == "1484"][0]
    assert rec_1484["is_diagnostic"] == "True"
    assert rec_1484["mask_status"] == "diagnostic_continuous_grayscale"


def test_10_native_record_60841288_remains_eligible(v2_rows):
    """Test 10: Native record 60841288 remains eligible native-target reference."""
    rec_60841288 = [r for r in v2_rows if r["cohort"] == "native_target" and r["pair_id"] == "60841288"][0]
    assert rec_60841288["is_primary_reference"] == "True"
    assert rec_60841288["is_diagnostic"] == "False"
    assert rec_60841288["mask_status"] == "canonical_binarized_jpeg"


def test_11_healthy_controls_outside_segmentation_training():
    """Test 11: Healthy controls remain outside segmentation-loss training."""
    loaders = build_stage_4_dataloaders()
    train_ds = loaders["train_dataset"]
    train_cohorts = set(r["cohort"] for r in train_ds.records)
    assert train_cohorts == {"source_training"}
    assert "healthy_controls" not in train_cohorts


def test_12_primary_metrics_count_unique_images():
    """Test 12: Primary metrics count unique images, not annotation records."""
    loaders = build_stage_4_dataloaders()
    target_ds = loaders["target_native_dataset"]
    assert len(target_ds) == 2187
    unique_target_imgs = len(set(r["orig_img_sha256"] for r in target_ds.records))
    assert unique_target_imgs == 2187


def test_13_loader_reads_corrected_v2_roles():
    """Test 13: The loader reads corrected v2 roles rather than stale v1 flags."""
    loaders = build_stage_4_dataloaders()
    target_ds = loaders["target_native_dataset"]
    # If stale v1 flags were read, len(target_ds) would be 2363. Under v2, it is exactly 2187.
    assert len(target_ds) == 2187


def test_14_original_v1_and_canonical_manifests_unchanged():
    """Test 14: Original v1 manifest and original Stage 3.2B canonical manifest remain unchanged."""
    assert compute_file_sha256(ORIG_MANIFEST_PATH) == EXPECTED_ORIG_MANIFEST_SHA256
    assert compute_file_sha256(V1_MANIFEST_PATH) == EXPECTED_V1_MANIFEST_SHA256

