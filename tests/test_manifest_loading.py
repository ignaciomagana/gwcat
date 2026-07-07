"""Tests for the declarative manifest loader (gwcat.manifests, PR 7).

Covers:
  * every bundled manifest (releases/ + injections/) loads and validates
  * a tiny user-supplied manifest loads from a filesystem path and produces
    a working file filter with zero code changes
  * validate_manifest() raises clear, file/field-naming errors on a broken
    manifest (missing required field) and on assorted malformed inputs
  * get_manifest() alias resolution and unknown-name error
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gwcat import manifests as m

FIXTURES = Path(__file__).parent / "fixtures"


def _valid_raw():
    """A minimal, fully valid raw manifest dict for mutation-based tests."""
    return {
        "release": "GWTC-TEST",
        "observing_runs": ["O-test"],
        "description": "test manifest",
        "records": [
            {"provider": "zenodo", "record_id": 123, "concept_id": 456},
        ],
        "products": {
            "pe_samples": {
                "file_patterns": ["*.h5"],
            },
        },
        "metadata": {
            "source_class_reference": "release_table",
            "far_available_online": False,
            "preferred_waveform_source": "manifest",
        },
        "validation": {
            "require_checksums": False,
            "allow_missing_far": True,
        },
    }


# ---------------------------------------------------------------------------
# Bundled manifests
# ---------------------------------------------------------------------------
class TestBundledManifests:
    def test_list_releases_nonempty(self):
        names = m.list_releases()
        assert names
        assert set(names) == set(m.list_release_manifests()) | set(
            m.list_injection_manifests()
        )

    def test_every_bundled_release_manifest_loads(self):
        for name in m.list_release_manifests():
            manifest = m.get_manifest(name)
            assert isinstance(manifest, m.ReleaseManifest)
            assert manifest.release == name
            assert manifest.records
            assert manifest.products

    def test_every_bundled_injection_manifest_loads(self):
        for name in m.list_injection_manifests():
            manifest = m.get_manifest(name)
            assert isinstance(manifest, m.ReleaseManifest)
            assert manifest.release == name
            assert manifest.records
            assert manifest.products

    def test_bundled_manifests_have_positive_record_ids(self):
        for name in m.list_releases():
            manifest = m.get_manifest(name)
            for rid in manifest.record_ids:
                assert isinstance(rid, int) and rid > 0
            for cid in manifest.concept_ids:
                assert cid is None or (isinstance(cid, int) and cid > 0)

    def test_gwtc4_alias_resolves_to_gwtc4p1(self):
        canonical = m.get_manifest("GWTC-4.1")
        alias = m.get_manifest("GWTC-4")
        # get_manifest() re-reads bundled YAML on each call (no caching), so
        # canonical/alias lookups return equal-content objects rather than
        # the same instance; what matters is they resolve to the same release.
        assert alias == canonical
        assert alias.release == "GWTC-4.1"
        assert "GWTC-4" in alias.aliases

    def test_observing_run_label_override(self):
        cumulative = m.get_manifest("injections-O1O2O3O4")
        assert cumulative.observing_run == "O1-O4b"

    def test_observing_run_defaults_to_join(self):
        gwtc21 = m.get_manifest("GWTC-2.1")
        assert gwtc21.observing_run_label is None
        assert gwtc21.observing_run == "+".join(gwtc21.observing_runs)
        assert gwtc21.observing_run == "O1+O2+O3a"

    def test_unknown_manifest_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown manifest"):
            m.get_manifest("GWTC-DOES-NOT-EXIST")


# ---------------------------------------------------------------------------
# User-supplied manifest from a filesystem path
# ---------------------------------------------------------------------------
class TestUserSuppliedManifest:
    def test_fake_manifest_loads_from_path(self):
        path = FIXTURES / "fake_release_manifest.yaml"
        manifest = m.get_manifest(str(path))
        assert manifest.release == "GWTC-FAKE"
        assert manifest.observing_runs == ["O-test"]
        assert manifest.record_ids == [999999]
        assert manifest.concept_ids == [999998]

    def test_fake_manifest_produces_working_file_filter(self):
        path = FIXTURES / "fake_release_manifest.yaml"
        manifest = m.load_manifest_file(path)
        product = manifest.products["pe_samples"]

        # A plausible cosmo PE file for a fake event should match.
        assert product.matches("FAKE_GW991231_235959_cosmo.h5")
        # nocosmo files must be rejected.
        assert not product.matches("FAKE_GW991231_235959_nocosmo.h5")
        # Files without an event name must be rejected.
        assert not product.matches("FAKE_summary_cosmo.h5")

    def test_load_manifest_file_directly(self):
        path = FIXTURES / "fake_release_manifest.yaml"
        manifest = m.load_manifest_file(path)
        assert manifest.source_path == str(path)


# ---------------------------------------------------------------------------
# validate_manifest() — broken fixture + targeted mutations
# ---------------------------------------------------------------------------
class TestValidation:
    def test_broken_manifest_missing_records_field(self):
        path = FIXTURES / "broken_manifest_missing_field.yaml"
        with pytest.raises(m.ManifestValidationError) as exc:
            m.load_manifest_file(path)
        msg = str(exc.value)
        assert str(path) in msg
        assert "records" in msg

    def test_valid_raw_passes(self):
        m.validate_manifest(_valid_raw(), source="test")

    @pytest.mark.parametrize(
        "key", ["release", "observing_runs", "description", "records",
                "products", "metadata", "validation"]
    )
    def test_missing_top_level_field_raises(self, key):
        raw = _valid_raw()
        del raw[key]
        with pytest.raises(m.ManifestValidationError) as exc:
            m.validate_manifest(raw, source="unit-test.yaml")
        assert "unit-test.yaml" in str(exc.value)
        assert key in str(exc.value)

    def test_wrong_type_release_raises(self):
        raw = _valid_raw()
        raw["release"] = 123
        with pytest.raises(m.ManifestValidationError, match="release"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_empty_observing_runs_raises(self):
        raw = _valid_raw()
        raw["observing_runs"] = []
        with pytest.raises(m.ManifestValidationError, match="observing_runs"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_empty_records_raises(self):
        raw = _valid_raw()
        raw["records"] = []
        with pytest.raises(m.ManifestValidationError, match="records"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_negative_record_id_raises(self):
        raw = _valid_raw()
        raw["records"][0]["record_id"] = -1
        with pytest.raises(m.ManifestValidationError, match="record_id"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_empty_products_raises(self):
        raw = _valid_raw()
        raw["products"] = {}
        with pytest.raises(m.ManifestValidationError, match="products"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_product_missing_file_patterns_raises(self):
        raw = _valid_raw()
        del raw["products"]["pe_samples"]["file_patterns"]
        with pytest.raises(m.ManifestValidationError, match="file_patterns"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_product_empty_file_patterns_raises(self):
        raw = _valid_raw()
        raw["products"]["pe_samples"]["file_patterns"] = []
        with pytest.raises(m.ManifestValidationError, match="file_patterns"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_metadata_missing_field_raises(self):
        raw = _valid_raw()
        del raw["metadata"]["far_available_online"]
        with pytest.raises(m.ManifestValidationError, match="far_available_online"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_validation_wrong_type_raises(self):
        raw = _valid_raw()
        raw["validation"]["allow_missing_far"] = "yes"
        with pytest.raises(m.ManifestValidationError, match="allow_missing_far"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_alias_matching_release_name_raises(self):
        raw = _valid_raw()
        raw["aliases"] = ["GWTC-TEST"]
        with pytest.raises(m.ManifestValidationError, match="aliases"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_blank_observing_run_label_raises(self):
        raw = _valid_raw()
        raw["observing_run_label"] = "   "
        with pytest.raises(m.ManifestValidationError, match="observing_run_label"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_dash_count_negative_raises(self):
        raw = _valid_raw()
        raw["products"]["pe_samples"]["dash_count"] = -1
        with pytest.raises(m.ManifestValidationError, match="dash_count"):
            m.validate_manifest(raw, source="unit-test.yaml")

    def test_valid_raw_unmutated(self):
        raw = _valid_raw()
        before = copy.deepcopy(raw)
        m.validate_manifest(raw, source="test")
        assert raw == before
