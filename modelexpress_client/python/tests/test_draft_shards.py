# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for engine-neutral draft shard selection."""

import json
import os

from modelexpress.draft_shards import (
    CONFIG_JSON_NAME,
    SAFETENSORS_INDEX_NAME,
    STATIC_DRAFT_PREFIXES,
    DraftShardSelection,
    draft_prefixes_for,
    extra_layer_draft_prefixes,
    narrow_to_draft_shards,
    select_draft_weight_files,
)


class TestExtraLayerDraftPrefixes:
    def test_derives_layers_past_the_target(self):
        prefixes = extra_layer_draft_prefixes(
            {"num_hidden_layers": 61, "num_nextn_predict_layers": 1}
        )
        assert "model.layers.61." in prefixes
        assert "model.layers.60." not in prefixes
        assert "model.layers.62." not in prefixes

    def test_covers_every_predict_layer(self):
        prefixes = extra_layer_draft_prefixes(
            {"num_hidden_layers": 78, "num_nextn_predict_layers": 2}
        )
        assert "model.layers.78." in prefixes
        assert "model.layers.79." in prefixes

    def test_reads_text_config_for_multimodal_checkpoints(self):
        prefixes = extra_layer_draft_prefixes(
            {
                "architectures": ["Glm5NextForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 40,
                    "num_nextn_predict_layers": 1,
                },
            }
        )
        assert "model.language_model.layers.40." in prefixes
        assert "model.layers.40." in prefixes

    def test_rejects_missing_zero_and_bool_counts(self):
        assert extra_layer_draft_prefixes(None) == ()
        assert extra_layer_draft_prefixes({}) == ()
        assert extra_layer_draft_prefixes({"num_hidden_layers": 61}) == ()
        assert (
            extra_layer_draft_prefixes(
                {"num_hidden_layers": 61, "num_nextn_predict_layers": 0}
            )
            == ()
        )
        assert (
            extra_layer_draft_prefixes(
                {"num_hidden_layers": True, "num_nextn_predict_layers": 1}
            )
            == ()
        )
        assert (
            extra_layer_draft_prefixes(
                {"num_hidden_layers": "61", "num_nextn_predict_layers": 1}
            )
            == ()
        )

    def test_static_prefixes_always_included(self):
        prefixes = draft_prefixes_for(None)
        assert prefixes == STATIC_DRAFT_PREFIXES
        assert "mtp." in prefixes
        assert "model.mtp." in prefixes


def _files(directory, names):
    return [os.path.join(directory, name) for name in names]


class TestSelectDraftWeightFiles:
    def test_selects_extra_layer_shards_only(self, tmp_path):
        index = {
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                "model.layers.0.mlp.weight": "model-00002-of-00004.safetensors",
                "model.layers.61.eh_proj.weight": "model-00003-of-00004.safetensors",
                "model.layers.61.enorm.weight": "model-00004-of-00004.safetensors",
                "lm_head.weight": "model-00004-of-00004.safetensors",
            }
        }
        files = _files(
            str(tmp_path),
            [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
        )
        prefixes = draft_prefixes_for(
            {"num_hidden_layers": 61, "num_nextn_predict_layers": 1}
        )
        assert select_draft_weight_files(index, files, prefixes, str(tmp_path)) == (
            DraftShardSelection.SELECTED,
            files[2:],
        )

    def test_layer_index_prefix_does_not_match_a_longer_index(self, tmp_path):
        # "model.layers.6." must not match layer 61 and vice versa.
        index = {
            "weight_map": {
                "model.layers.6.mlp.weight": "base.safetensors",
                "model.layers.61.eh_proj.weight": "draft.safetensors",
            }
        }
        files = _files(str(tmp_path), ["base.safetensors", "draft.safetensors"])
        prefixes = draft_prefixes_for(
            {"num_hidden_layers": 61, "num_nextn_predict_layers": 1}
        )
        assert select_draft_weight_files(index, files, prefixes, str(tmp_path)) == (
            DraftShardSelection.SELECTED,
            [files[1]],
        )

    def test_selects_mtp_prefixed_shards_without_config(self, tmp_path):
        index = {
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "mtp.fc.weight": "model-mtp.safetensors",
            }
        }
        files = _files(
            str(tmp_path),
            ["model-00001-of-00002.safetensors", "model-mtp.safetensors"],
        )
        assert select_draft_weight_files(
            index, files, draft_prefixes_for(None), str(tmp_path)
        ) == (DraftShardSelection.SELECTED, [files[1]])

    def test_matches_subfolder_shards_by_relative_path(self, tmp_path):
        index = {"weight_map": {"mtp.fc.weight": "draft/model.safetensors"}}
        files = [
            os.path.join(str(tmp_path), "base", "model.safetensors"),
            os.path.join(str(tmp_path), "draft", "model.safetensors"),
        ]
        assert select_draft_weight_files(
            index, files, draft_prefixes_for(None), str(tmp_path)
        ) == (DraftShardSelection.SELECTED, [files[1]])

    def test_no_draft_tensors_in_index(self, tmp_path):
        index = {"weight_map": {"model.embed_tokens.weight": "a.safetensors"}}
        files = _files(str(tmp_path), ["a.safetensors"])
        assert select_draft_weight_files(
            index, files, draft_prefixes_for(None), str(tmp_path)
        ) == (DraftShardSelection.NO_DRAFT_WEIGHTS, [])

    def test_missing_index_is_unresolved(self, tmp_path):
        files = _files(str(tmp_path), ["a.safetensors"])
        assert select_draft_weight_files(
            None, files, draft_prefixes_for(None), str(tmp_path)
        ) == (DraftShardSelection.UNRESOLVED, [])

    def test_index_naming_shards_not_on_disk_is_unresolved(self, tmp_path):
        index = {"weight_map": {"mtp.fc.weight": "elsewhere.safetensors"}}
        files = _files(str(tmp_path), ["a.safetensors"])
        assert select_draft_weight_files(
            index, files, draft_prefixes_for(None), str(tmp_path)
        ) == (DraftShardSelection.UNRESOLVED, [])

    def test_empty_prefixes_never_select(self, tmp_path):
        index = {"weight_map": {"mtp.fc.weight": "a.safetensors"}}
        files = _files(str(tmp_path), ["a.safetensors"])
        assert select_draft_weight_files(index, files, (), str(tmp_path)) == (
            DraftShardSelection.NO_DRAFT_WEIGHTS,
            [],
        )


class TestNarrowToDraftShards:
    def _write(self, tmp_path, index, config):
        (tmp_path / SAFETENSORS_INDEX_NAME).write_text(
            json.dumps(index), encoding="utf-8"
        )
        if config is not None:
            (tmp_path / CONFIG_JSON_NAME).write_text(
                json.dumps(config), encoding="utf-8"
            )

    def test_narrows_using_on_disk_config(self, tmp_path, caplog):
        self._write(
            tmp_path,
            {
                "weight_map": {
                    "model.layers.0.w": "model-00001-of-00003.safetensors",
                    "model.layers.78.w": "model-00002-of-00003.safetensors",
                    "model.layers.78.embed_tokens.weight": (
                        "model-00003-of-00003.safetensors"
                    ),
                }
            },
            {"num_hidden_layers": 78, "num_nextn_predict_layers": 1},
        )
        files = _files(
            str(tmp_path), [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
        )
        with caplog.at_level("INFO", logger="modelexpress.draft_shards"):
            narrowed = narrow_to_draft_shards(str(tmp_path), files, log_prefix="[W] ")
        assert narrowed == files[1:]
        assert any("loading 2 of 3" in rec.message for rec in caplog.records)

    def test_keeps_all_shards_without_draft_head(self, tmp_path, caplog):
        self._write(
            tmp_path,
            {"weight_map": {"model.layers.0.w": "a.safetensors"}},
            {"num_hidden_layers": 4},
        )
        files = _files(str(tmp_path), ["a.safetensors"])
        with caplog.at_level("INFO", logger="modelexpress.draft_shards"):
            assert narrow_to_draft_shards(str(tmp_path), files) == files
        assert any("holds no draft tensors" in rec.message for rec in caplog.records)

    def test_keeps_all_shards_without_index(self, tmp_path, caplog):
        files = _files(str(tmp_path), ["a.safetensors"])
        with caplog.at_level("WARNING", logger="modelexpress.draft_shards"):
            assert narrow_to_draft_shards(str(tmp_path), files) == files
        assert any("could not resolve" in rec.message for rec in caplog.records)

    def test_keeps_all_shards_on_corrupt_index(self, tmp_path):
        (tmp_path / SAFETENSORS_INDEX_NAME).write_text("{not json", encoding="utf-8")
        files = _files(str(tmp_path), ["a.safetensors"])
        assert narrow_to_draft_shards(str(tmp_path), files) == files

    def test_keeps_all_shards_for_object_store_folder(self):
        files = ["s3://bucket/model/a.safetensors"]
        assert narrow_to_draft_shards("s3://bucket/model", files) == files

    def test_runtime_config_is_not_consulted(self, tmp_path):
        # Only the on-disk config derives layer prefixes. Without config.json
        # the static prefixes still apply.
        self._write(
            tmp_path,
            {
                "weight_map": {
                    "model.layers.0.w": "a.safetensors",
                    "mtp.fc.weight": "b.safetensors",
                }
            },
            None,
        )
        files = _files(str(tmp_path), ["a.safetensors", "b.safetensors"])
        assert narrow_to_draft_shards(str(tmp_path), files) == [files[1]]
