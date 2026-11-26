"""
Focused tests for the production Stage 4 Dataset / DataLoader integration.

Synthetic fixtures use tiny images under temporary directories. Real-data tests
read the authoritative corrected-v2 manifest and Task 3B field manifest without
copying or modifying them.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.boundary_fields import (
    FIELD_VERSION,
    HASH_CONVENTION,
    MAX_CAP,
    TAU,
    canonical_field_hash,
)
from src.data.samplers import EpochShuffleSampler
from src.data.stage_4_dataloaders import (
    DFUCanonicalDataset,
    build_stage_4_dataloaders,
)
from src.data.transforms import (
    compute_tolerance_boundary_fields,
    derive_augmentation_seed,
)

REAL_ROOT = Path("E:/DFU_Cross_Domain_Research")
REAL_MANIFEST = (
    "data/processed/stage_3_2B_manifest_repair_v2/"
    "canonical_manifest_corrected_v2.csv"
)
REAL_FIELD_DIR = "data/processed/stage_4A_boundary_fields_v1"
REAL_CANONICAL_DIR = "data/processed/stage_3_2B_canonical"
SIZE = 16


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _spec(
    cohort: str = "source_training",
    pair_id: str = "0001",
    mask: np.ndarray | None = None,
    *,
    is_primary: bool = True,
    is_diagnostic: bool = False,
    cluster_id: str | None = None,
    image_sha: str | None = None,
    mask_status: str = "strict_binary",
    mask_filename: str | None = None,
) -> dict:
    if mask is None:
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[5:11, 5:11] = True
    return {
        "cohort": cohort,
        "pair_id": pair_id,
        "mask": mask,
        "is_primary": is_primary,
        "is_diagnostic": is_diagnostic,
        "cluster_id": cluster_id or f"{cohort}_cluster_{pair_id}",
        "image_sha": image_sha or f"{cohort}_imgsha_{pair_id}",
        "mask_status": mask_status,
        "mask_filename": mask_filename,
    }


def _deterministic_image(size: int = SIZE) -> np.ndarray:
    ramp = (np.arange(size * size).reshape(size, size) % 256).astype(np.uint8)
    image = np.stack([ramp, np.flipud(ramp), np.rot90(ramp)], axis=-1)
    image[0, 0] = np.array([255, 255, 255], dtype=np.uint8)
    return image


def _build_tree(tmp_path: Path, specs: list[dict], size: int = SIZE) -> dict:
    canonical = tmp_path / "canonical"
    images_dir = canonical / "images"
    masks_dir = canonical / "masks"
    fields = tmp_path / "fields"
    (fields / "phi_fp").mkdir(parents=True, exist_ok=True)
    (fields / "phi_fn").mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    field_rows: list[dict] = []
    image_by_record: dict[str, np.ndarray] = {}

    for spec in specs:
        cohort = spec["cohort"]
        pair_id = spec["pair_id"]
        record_id = f"{cohort}_{pair_id}"
        mask = spec["mask"]
        image = _deterministic_image(size)
        image_name = f"{cohort}_{pair_id}_bilinear.png"
        image_path = images_dir / image_name
        Image.fromarray(image, mode="RGB").save(image_path, format="PNG")
        image_by_record[record_id] = image

        mask_filename = (
            spec["mask_filename"] or f"{cohort}_{pair_id}_canonical_mask.png"
        )
        mask_path = masks_dir / mask_filename
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
            mask_path, format="PNG"
        )
        mask_sha = _sha256(mask_path)
        image_sha_file = _sha256(image_path)

        phi_fp, phi_fn = compute_tolerance_boundary_fields(
            mask, tau=TAU, max_cap=MAX_CAP
        )
        np.save(fields / "phi_fp" / f"{record_id}.npy", phi_fp, allow_pickle=False)
        np.save(fields / "phi_fn" / f"{record_id}.npy", phi_fn, allow_pickle=False)

        manifest_rows.append(
            {
                "cohort": cohort,
                "pair_id": pair_id,
                "cluster_id": spec["cluster_id"],
                "unique_content_id": spec["image_sha"],
                "is_primary_reference": str(spec["is_primary"]),
                "is_diagnostic": str(spec["is_diagnostic"]),
                "orig_img_path": f"orig/{image_name}",
                "orig_img_w": str(size),
                "orig_img_h": str(size),
                "orig_img_sha256": spec["image_sha"],
                "proc_img_bilinear_sha256": image_sha_file,
                "proc_mask_filename": mask_filename,
                "proc_mask_sha256": mask_sha,
                "mask_status": spec["mask_status"],
                "orig_fg_count": str(int(mask.sum())),
                "proc_fg_count": str(int(mask.sum())),
                "phi_fp_sha256": hashlib.sha256(phi_fp.tobytes()).hexdigest(),
                "phi_fn_sha256": hashlib.sha256(phi_fn.tobytes()).hexdigest(),
            }
        )
        field_rows.append(
            {
                "record_id": record_id,
                "pair_id": pair_id,
                "cohort": cohort,
                "annotation_role": "primary" if spec["is_primary"] else "alternate",
                "cluster_id": spec["cluster_id"],
                "unique_content_id": spec["image_sha"],
                "unit_role": "primary_unique_image",
                "is_independent_training_unit": "True",
                "source_mask_path": mask_path.as_posix(),
                "source_mask_sha256_manifest": mask_sha,
                "source_mask_sha256_actual": mask_sha,
                "source_mask_hash_verified": "True",
                "field_algorithm": "compute_tolerance_boundary_fields",
                "field_version": FIELD_VERSION,
                "hash_convention": HASH_CONVENTION,
                "tau": str(TAU),
                "max_cap": str(MAX_CAP),
                "output_dtype": "float32",
                "output_shape": f"{size}x{size}",
                "phi_fp_filename": f"{record_id}.npy",
                "phi_fn_filename": f"{record_id}.npy",
                "phi_fp_content_hash": canonical_field_hash(phi_fp),
                "phi_fn_content_hash": canonical_field_hash(phi_fn),
                "phi_fp_file_sha256": _sha256(fields / "phi_fp" / f"{record_id}.npy"),
                "phi_fn_file_sha256": _sha256(fields / "phi_fn" / f"{record_id}.npy"),
                "is_empty_mask": str(not np.any(mask)),
                "foreground_pixel_count": str(int(mask.sum())),
                "interior_field_all_zero": str(
                    bool(np.any(mask)) and not bool(np.any(phi_fn))
                ),
                "is_diagnostic": str(spec["is_diagnostic"]),
                "generation_status": "generated",
            }
        )

    manifest_path = tmp_path / "canonical_manifest.csv"
    field_manifest_path = fields / "boundary_field_manifest.csv"
    _write_csv(manifest_path, manifest_rows)
    _write_csv(field_manifest_path, field_rows)

    return {
        "manifest": manifest_path,
        "field_manifest": field_manifest_path,
        "canonical": canonical,
        "fields": fields,
        "image_by_record": image_by_record,
        "manifest_rows": manifest_rows,
        "field_rows": field_rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty fixture CSV")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(
    tree: dict,
    cohort: str = "source_training",
    *,
    expected_count: int | None = None,
    verify_field_hashes: bool = True,
) -> DFUCanonicalDataset:
    return DFUCanonicalDataset(
        manifest_path=tree["manifest"],
        cohort=cohort,
        project_root=tree["manifest"].parent,
        canonical_dir=tree["canonical"],
        boundary_field_dir=tree["fields"],
        expected_count=expected_count,
        verify_field_hashes=verify_field_hashes,
    )


# ---------------------------------------------------------------------------
# 1–5. Dataset contract
# ---------------------------------------------------------------------------
def test_dataset_is_real_torch_dataset(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    ds = _make_dataset(tree)
    assert isinstance(ds, Dataset)
    assert isinstance(ds, torch.utils.data.Dataset)
    assert len(ds) == 1


def test_image_is_raw_rgb_channel_first_without_normalization(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec(cohort="internal_validation", pair_id="0001")])
    ds = _make_dataset(tree, cohort="internal_validation")
    sample = ds[0]
    image = sample["image"]

    assert image.shape == (3, SIZE, SIZE)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0

    expected = (
        torch.from_numpy(tree["image_by_record"]["internal_validation_0001"])
        .permute(2, 0, 1)
        .to(torch.float32)
        / 255.0
    )
    # Exact raw-RGB reproduction: no ImageNet normalization in the Dataset.
    assert torch.equal(image, expected)
    assert float(image.max()) > 0.9


def test_mask_is_binary_single_channel(tmp_path: Path):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[4:12, 3:9] = True
    tree = _build_tree(tmp_path, [_spec(mask=mask)])
    sample = _make_dataset(tree)[0]

    assert sample["mask"].shape == (1, SIZE, SIZE)
    assert sample["mask"].dtype == torch.float32
    values = set(np.unique(sample["mask"].numpy()).tolist())
    assert values.issubset({0.0, 1.0})
    assert int(sample["mask"].sum()) == int(mask.sum())


def test_mask_filename_is_not_used_to_infer_emptiness(tmp_path: Path):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[2:10, 2:10] = True
    tree = _build_tree(
        tmp_path,
        [_spec(mask=mask, mask_filename="source_training_0001_empty_mask.png")],
    )
    sample = _make_dataset(tree)[0]
    assert int(sample["mask"].sum()) > 0


def test_boundary_fields_are_single_channel_float32(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    sample = _make_dataset(tree)[0]
    for key in ("phi_fp", "phi_fn"):
        assert sample[key].shape == (1, SIZE, SIZE)
        assert sample[key].dtype == torch.float32
        assert torch.isfinite(sample[key]).all()


# ---------------------------------------------------------------------------
# 6–10. Boundary-field integrity
# ---------------------------------------------------------------------------
def test_valid_canonical_hash_is_accepted(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec(cohort="internal_validation", pair_id="0001")])
    ds = _make_dataset(tree, cohort="internal_validation")
    sample = ds[0]
    recomputed = canonical_field_hash(sample["phi_fp"][0].numpy())
    assert recomputed == sample["phi_fp_content_hash"]


def test_tampered_field_raises_clear_hash_error(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    ds = _make_dataset(tree)
    fp_path = tree["fields"] / "phi_fp" / "source_training_0001.npy"
    tampered = np.zeros((SIZE, SIZE), dtype=np.float32)
    np.save(fp_path, tampered, allow_pickle=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        ds[0]


def test_wrong_shape_and_dtype_are_rejected(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    ds = _make_dataset(tree)
    fp_path = tree["fields"] / "phi_fp" / "source_training_0001.npy"

    np.save(fp_path, np.zeros((SIZE // 2, SIZE // 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="shape"):
        ds[0]

    np.save(fp_path, np.zeros((SIZE, SIZE), dtype=np.float64), allow_pickle=False)
    with pytest.raises(ValueError, match="dtype"):
        ds[0]


def test_non_finite_field_values_are_rejected(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    ds = _make_dataset(tree)
    fn_path = tree["fields"] / "phi_fn" / "source_training_0001.npy"
    bad = np.zeros((SIZE, SIZE), dtype=np.float32)
    bad[0, 0] = np.nan
    np.save(fn_path, bad, allow_pickle=False)
    with pytest.raises(ValueError, match="non-finite"):
        ds[0]


@pytest.mark.parametrize("artifact", ["image", "mask", "phi_fp", "phi_fn"])
def test_missing_artifact_raises_clear_error(tmp_path: Path, artifact: str):
    tree = _build_tree(tmp_path, [_spec()])
    if artifact == "image":
        path = tree["canonical"] / "images" / "source_training_0001_bilinear.png"
    elif artifact == "mask":
        path = tree["canonical"] / "masks" / "source_training_0001_canonical_mask.png"
    elif artifact == "phi_fp":
        path = tree["fields"] / "phi_fp" / "source_training_0001.npy"
    else:
        path = tree["fields"] / "phi_fn" / "source_training_0001.npy"
    path.unlink()

    with pytest.raises(FileNotFoundError, match="Missing Stage 4 artifacts"):
        _make_dataset(tree)


def test_field_removed_after_construction_raises_on_access(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    ds = _make_dataset(tree)
    (tree["fields"] / "phi_fp" / "source_training_0001.npy").unlink()
    with pytest.raises(RuntimeError, match="Failed to load phi_fp"):
        ds[0]


# ---------------------------------------------------------------------------
# 11–15. Annotation / cohort enforcement
# ---------------------------------------------------------------------------
def test_primary_selection_uses_lowest_pair_id(tmp_path: Path):
    specs = [
        _spec(
            pair_id="0002",
            is_primary=False,
            image_sha="shared_hash",
            cluster_id="shared_cluster",
        ),
        _spec(
            pair_id="0001",
            is_primary=True,
            image_sha="shared_hash",
            cluster_id="shared_cluster",
        ),
    ]
    tree = _build_tree(tmp_path, specs)
    ds = _make_dataset(tree)
    assert len(ds) == 1
    assert [r["pair_id"] for r in ds.records] == ["0001"]


def test_wrong_primary_flag_for_lowest_pair_id_is_rejected(tmp_path: Path):
    specs = [
        _spec(
            pair_id="0001",
            is_primary=False,
            image_sha="shared_hash",
            cluster_id="shared_cluster",
        ),
        _spec(
            pair_id="0002",
            is_primary=True,
            image_sha="shared_hash",
            cluster_id="shared_cluster",
        ),
    ]
    tree = _build_tree(tmp_path, specs)
    with pytest.raises(ValueError, match="does not match the lowest"):
        _make_dataset(tree)


def test_multiple_primaries_in_cluster_are_rejected(tmp_path: Path):
    specs = [
        _spec(
            pair_id="0001",
            is_primary=True,
            image_sha="shared_hash",
            cluster_id="shared_cluster",
        ),
        _spec(
            pair_id="0002",
            is_primary=True,
            image_sha="shared_hash",
            cluster_id="shared_cluster",
        ),
    ]
    tree = _build_tree(tmp_path, specs)
    with pytest.raises(ValueError, match="has 2 primary records"):
        _make_dataset(tree)


def test_alternates_never_enter_dataset_or_sampler(tmp_path: Path):
    specs = [
        _spec(pair_id="0001", image_sha="h1", cluster_id="c1"),
        _spec(pair_id="0002", is_primary=False, image_sha="h1", cluster_id="c1"),
        _spec(pair_id="0003", image_sha="h2", cluster_id="c2"),
        _spec(pair_id="0004", is_primary=False, image_sha="h2", cluster_id="c2"),
        _spec(pair_id="0005", image_sha="h3", cluster_id="c3"),
    ]
    tree = _build_tree(tmp_path, specs)
    ds = _make_dataset(tree, expected_count=3)

    assert len(ds) == 3
    assert {r["pair_id"] for r in ds.records} == {"0001", "0003", "0005"}
    sampler = EpochShuffleSampler(len(ds), seed=42)
    indices = list(sampler)
    assert len(indices) == 3
    assert len(set(indices)) == 3
    assert sorted(indices) == [0, 1, 2]


def test_diagnostic_records_excluded(tmp_path: Path):
    specs = [
        _spec(pair_id="0001", image_sha="h1"),
        _spec(pair_id="0002", image_sha="h2"),
        _spec(
            pair_id="0233",
            image_sha="h3",
            is_diagnostic=True,
            mask_status="diagnostic_multiclass",
        ),
    ]
    tree = _build_tree(tmp_path, specs)
    ds = _make_dataset(tree, expected_count=2)
    assert len(ds) == 2
    assert "0233" not in {r["pair_id"] for r in ds.records}
    assert ds.excluded_diagnostic_count == 1


def test_healthy_controls_are_rejected_as_segmentation_dataset(tmp_path: Path):
    specs = [_spec(cohort="healthy_controls")]
    tree = _build_tree(tmp_path, specs)
    with pytest.raises(ValueError, match="not a segmentation cohort"):
        _make_dataset(tree, cohort="healthy_controls")


# ---------------------------------------------------------------------------
# 16–18. Real-manifest population counts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_loaders():
    return build_stage_4_dataloaders(
        project_root=REAL_ROOT,
        manifest_path=REAL_MANIFEST,
        canonical_dir=REAL_CANONICAL_DIR,
        boundary_field_dir=REAL_FIELD_DIR,
        batch_size=2,
        num_workers=0,
        seed=42,
    )


def test_real_train_dataset_exact_count(real_loaders):
    ds = real_loaders["train_dataset"]
    assert len(ds) == 945
    assert len({r["pair_id"] for r in ds.records}) == 945
    assert len({r["orig_img_sha256"] for r in ds.records}) == 945
    assert all(r["cohort"] == "source_training" for r in ds.records)
    assert all(r["annotation_role"] == "primary" for r in ds.records)


def test_real_validation_dataset_exact_count(real_loaders):
    ds = real_loaders["val_dataset"]
    assert len(ds) == 191
    assert all(r["cohort"] == "internal_validation" for r in ds.records)
    assert "0233" not in {r["pair_id"] for r in ds.records}


def test_real_native_target_dataset_exact_count(real_loaders):
    ds = real_loaders["target_native_dataset"]
    assert len(ds) == 2187
    assert len({r["orig_img_sha256"] for r in ds.records}) == 2187
    assert all(r["cohort"] == "native_target" for r in ds.records)


# ---------------------------------------------------------------------------
# 19–20. Sampler reproducibility and epoch behaviour
# ---------------------------------------------------------------------------
def test_sampler_reproducible_for_fixed_seed_and_epoch():
    a = EpochShuffleSampler(50, seed=42)
    b = EpochShuffleSampler(50, seed=42)
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)

    c = EpochShuffleSampler(50, seed=43)
    c.set_epoch(3)
    assert list(c) != list(a)


def test_sampler_changes_across_epochs_and_covers_all_indices():
    sampler = EpochShuffleSampler(50, seed=42)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == epoch0
    assert epoch0 != epoch1
    assert sorted(epoch0) == list(range(50))
    assert sorted(epoch1) == list(range(50))


def test_source_training_augmentation_seed_uses_record_and_epoch(monkeypatch, tmp_path):
    tree = _build_tree(tmp_path, [_spec(pair_id="0001")])
    ds = _make_dataset(tree)

    captured: dict = {}

    def _fake_augment(image, mask, seed=None):
        captured["seed"] = seed
        return image, mask

    import src.data.stage_4_dataloaders as stage4_module

    monkeypatch.setattr(stage4_module, "augment_sample", _fake_augment)

    ds.set_epoch(3)
    ds[0]
    assert captured["seed"] == derive_augmentation_seed(42, "source_training_0001", 3)

    ds.training_seed = 1337
    ds.set_epoch(4)
    ds[0]
    assert captured["seed"] == derive_augmentation_seed(1337, "source_training_0001", 4)


def test_validation_and_target_cohorts_are_not_augmented(monkeypatch, tmp_path):
    import src.data.stage_4_dataloaders as stage4_module

    calls: list = []

    def _fake_augment(image, mask, seed=None):
        calls.append(seed)
        return image, mask

    monkeypatch.setattr(stage4_module, "augment_sample", _fake_augment)

    val_tree = _build_tree(tmp_path / "val", [_spec(cohort="internal_validation", pair_id="0001")])
    val_ds = _make_dataset(val_tree, cohort="internal_validation")
    val_ds[0]

    target_tree = _build_tree(tmp_path / "target", [_spec(cohort="native_target", pair_id="0001")])
    target_ds = _make_dataset(target_tree, cohort="native_target")
    target_ds[0]

    assert calls == []


# ---------------------------------------------------------------------------
# 21–23. DataLoader batching, alignment, and metadata collation
# ---------------------------------------------------------------------------
def test_real_dataloader_batch_contract(real_loaders):
    batch = next(iter(real_loaders["train_loader"]))
    assert batch["image"].shape == (2, 3, 512, 512)
    assert batch["image"].dtype == torch.float32
    assert float(batch["image"].min()) >= 0.0
    assert float(batch["image"].max()) <= 1.0
    assert batch["mask"].shape == (2, 1, 512, 512)
    assert batch["mask"].dtype == torch.float32
    assert set(np.unique(batch["mask"].numpy()).tolist()).issubset({0.0, 1.0})
    assert batch["phi_fp"].shape == (2, 1, 512, 512)
    assert batch["phi_fn"].shape == (2, 1, 512, 512)
    assert torch.isfinite(batch["phi_fp"]).all()
    assert torch.isfinite(batch["phi_fn"]).all()

    # Metadata is collated into deterministic sequences.
    assert isinstance(batch["pair_id"], list) and len(batch["pair_id"]) == 2
    assert isinstance(batch["record_id"], list) and len(batch["record_id"]) == 2
    assert isinstance(batch["cohort"], list)
    assert set(batch["cohort"]) == {"source_training"}
    assert set(batch["annotation_role"]) == {"primary"}


def test_joint_alignment_of_image_mask_and_fields(tmp_path: Path):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[4:11, 6:13] = True
    tree = _build_tree(tmp_path, [_spec(mask=mask)])
    sample = _make_dataset(tree)[0]

    loaded_mask = sample["mask"][0].numpy().astype(bool)
    assert np.array_equal(loaded_mask, mask)
    assert sample["image"].shape[-2:] == sample["mask"].shape[-2:]
    assert sample["phi_fp"].shape[-2:] == sample["mask"].shape[-2:]

    phi_fp = sample["phi_fp"][0].numpy()
    phi_fn = sample["phi_fn"][0].numpy()
    # Same spatial convention: phi_fp is exterior-only, phi_fn is interior-only.
    assert not np.any(phi_fp[loaded_mask] > 0)
    assert not np.any(phi_fn[~loaded_mask] > 0)


def test_synthetic_dataloader_collates_batch(tmp_path: Path):
    specs = [
        _spec(pair_id="0001", image_sha="h1"),
        _spec(pair_id="0002", image_sha="h2"),
        _spec(pair_id="0003", image_sha="h3"),
    ]
    tree = _build_tree(tmp_path, specs)
    ds = _make_dataset(tree, expected_count=3)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, SIZE, SIZE)
    assert batch["mask"].shape == (2, 1, SIZE, SIZE)
    assert batch["phi_fp"].shape == (2, 1, SIZE, SIZE)
    assert batch["phi_fn"].shape == (2, 1, SIZE, SIZE)
    assert batch["pair_id"] == ["0001", "0002"]
    assert batch["is_primary_reference"].tolist() == [True, True]


# ---------------------------------------------------------------------------
# Additional integrity checks
# ---------------------------------------------------------------------------
def test_expected_count_mismatch_fails_loudly(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec(pair_id="0001")])
    with pytest.raises(ValueError, match="expected exactly"):
        _make_dataset(tree, expected_count=999)


def test_duplicate_field_records_are_rejected(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    rows = list(csv.DictReader(open(tree["field_manifest"], encoding="utf-8")))
    _write_csv(tree["field_manifest"], rows + [dict(rows[0])])
    with pytest.raises(ValueError, match="Duplicate boundary-field record_id"):
        _make_dataset(tree)


def test_stale_source_mask_hash_in_field_manifest_is_rejected(tmp_path: Path):
    tree = _build_tree(tmp_path, [_spec()])
    rows = list(csv.DictReader(open(tree["field_manifest"], encoding="utf-8")))
    rows[0]["source_mask_sha256_manifest"] = "0" * 64
    _write_csv(tree["field_manifest"], rows)
    with pytest.raises(ValueError, match="Stale boundary-field record"):
        _make_dataset(tree)


def test_real_factory_returns_real_dataloaders(real_loaders):
    assert isinstance(real_loaders["train_loader"], DataLoader)
    assert isinstance(real_loaders["val_loader"], DataLoader)
    assert isinstance(real_loaders["target_native_loader"], DataLoader)
    assert isinstance(real_loaders["train_sampler"], EpochShuffleSampler)
    assert real_loaders["train_loader"].batch_size == 2
    assert isinstance(
        real_loaders["val_loader"].sampler, torch.utils.data.SequentialSampler
    )
    assert isinstance(
        real_loaders["target_native_loader"].sampler,
        torch.utils.data.SequentialSampler,
    )
    assert "healthy_controls_dataset" not in real_loaders
    assert "target_mismatched_dataset" not in real_loaders