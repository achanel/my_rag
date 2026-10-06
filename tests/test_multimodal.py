"""Offline tests for chapter 4 (multimodal, modular RAG).

Ollama, the vision model and Deep Lake are replaced by small fakes, so these run
without network or models. ``collect_visdrone`` needs Activeloop and is not covered.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

from PIL import Image

from my_rag.cli import build_parser, main
from my_rag.config import Settings
from my_rag.llm import LLM
from my_rag.multimodal import (
    Sample,
    describe_objects,
    draw_boxes,
    image_data_url,
    load_samples,
    load_text_chunks,
    MultimodalRAG,
    target_label,
)

VOCABULARY = ("truck", "car", "pedestrian", "drones", "vehicles", "identify", "cameras")


def _bag_of_words(text: str) -> list[float]:
    words = re.findall(r"[a-z]+", text.lower())
    return [float(words.count(term)) for term in VOCABULARY]


class FakeEmbeddings:
    """Deterministic stand-in for the embedding model (counts of VOCABULARY words)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_bag_of_words(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeLLM(LLM):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, temperature=None) -> str:
        self.prompts.append(prompt)
        return "The drone spots trucks among the cars."


class _FakeCompletions:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


class FakeVisionClient:
    """Mimics ``OpenAI().chat.completions.create`` and records the request."""

    def __init__(self, reply: str = "  Yes, there is one truck in the boxes.  ") -> None:
        self.completions = _FakeCompletions(reply)
        self.chat = SimpleNamespace(completions=self.completions)


def _settings(**overrides) -> Settings:
    values = dict(
        provider="ollama",
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        embedding_model="all-minilm",
        vision_model="gemma3:12b",
    )
    values.update(overrides)
    return Settings(**values)


def _sample(id_: str, labels: list[str], path: Path | None = None) -> Sample:
    boxes = [[10.0 + 20 * index, 10.0, 15.0, 15.0] for index in range(len(labels))]
    return Sample(id=id_, path=path or Path(f"{id_}.jpg"), boxes=boxes, labels=labels)


class SampleTests(unittest.TestCase):
    def test_label_text_lists_distinct_classes_most_frequent_first(self):
        sample = _sample("00000", ["car", "truck", "car", "car", "truck", "pedestrian"])
        self.assertEqual(sample.label_text(), "car truck pedestrian")

    def test_image_without_boxes_gets_a_non_empty_placeholder(self):
        self.assertEqual(_sample("00000", []).label_text(), "no objects")


class TargetLabelTests(unittest.TestCase):
    def test_class_named_in_the_query_wins_over_a_commoner_class(self):
        sample = _sample("00000", ["car", "car", "car", "truck"])
        self.assertEqual(target_label("How do drones identify a truck?", sample), "truck")

    def test_plural_in_the_query_still_matches(self):
        sample = _sample("00000", ["car", "truck"])
        self.assertEqual(target_label("Can drones count trucks?", sample), "truck")

    def test_without_a_named_class_the_most_frequent_one_is_used(self):
        sample = _sample("00000", ["car", "pedestrian", "car"])
        self.assertEqual(target_label("How do drones identify vehicles?", sample), "car")

    def test_ignored_regions_and_others_are_never_the_target(self):
        sample = _sample("00000", ["ignored regions"] * 5 + ["others"] * 3 + ["truck"])
        self.assertEqual(target_label("How do drones find objects?", sample), "truck")

    def test_image_with_only_unannotated_objects_is_rejected(self):
        sample = _sample("00000", ["ignored regions", "others"])
        with self.assertRaises(ValueError):
            target_label("anything", sample)


class LoadSamplesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_annotations(self, records: list[dict]) -> None:
        (self.dir / "annotations.json").write_text(json.dumps(records), encoding="utf-8")

    def test_reads_annotations_and_resolves_image_paths(self):
        self._write_annotations(
            [{"id": "00000", "file": "00000.jpg", "boxes": [[1, 2, 3, 4]], "labels": ["truck"]}]
        )
        (sample,) = load_samples(self.dir)
        self.assertEqual(sample.id, "00000")
        self.assertEqual(sample.path, self.dir / "00000.jpg")
        self.assertEqual(sample.boxes, [[1, 2, 3, 4]])
        self.assertEqual(sample.labels, ["truck"])

    def test_boxes_and_labels_must_line_up(self):
        self._write_annotations(
            [{"id": "00000", "file": "00000.jpg", "boxes": [[1, 2, 3, 4]], "labels": ["truck", "car"]}]
        )
        with self.assertRaisesRegex(ValueError, "1 boxes but 2 labels"):
            load_samples(self.dir)

    def test_missing_annotations_point_to_the_collect_command(self):
        with self.assertRaisesRegex(FileNotFoundError, "--collect --corpus visdrone"):
            load_samples(self.dir)

    def test_empty_annotations_are_rejected(self):
        self._write_annotations([])
        with self.assertRaisesRegex(ValueError, "no samples"):
            load_samples(self.dir)

    def test_missing_text_corpus_points_to_the_drone_collect_command(self):
        with self.assertRaisesRegex(FileNotFoundError, "--collect --corpus drone"):
            load_text_chunks(self.dir / "drone.md")


class DrawBoxesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        Image.new("RGB", (100, 100), "white").save(self.dir / "source.png")
        self.sample = Sample(
            id="00000",
            path=self.dir / "source.png",
            boxes=[[10.0, 10.0, 40.0, 40.0], [60.0, 60.0, 20.0, 20.0]],
            labels=["truck", "car"],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_only_the_requested_class_is_outlined_in_red(self):
        output = draw_boxes(self.sample, "truck", self.dir / "out" / "boxed.jpg")
        self.assertTrue(output.is_file())
        image = Image.open(output).convert("RGB")
        # Left edge of the truck box (JPEG is lossy, so check for "clearly red").
        red = image.getpixel((11, 30))
        self.assertGreater(red[0], 180)
        self.assertLess(red[1], 120)
        self.assertLess(red[2], 120)
        # Left edge of the car box is not drawn when the target is the truck.
        untouched = image.getpixel((61, 70))
        self.assertTrue(all(channel > 200 for channel in untouched), untouched)


class ImageDataUrlTests(unittest.TestCase):
    def test_jpeg_bytes_round_trip_through_the_data_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.jpg"
            path.write_bytes(b"\xff\xd8\xff fake jpeg bytes")
            url = image_data_url(path)
        prefix = "data:image/jpeg;base64,"
        self.assertTrue(url.startswith(prefix))
        self.assertEqual(base64.b64decode(url[len(prefix):]), b"\xff\xd8\xff fake jpeg bytes")


class DescribeObjectsTests(unittest.TestCase):
    def test_sends_the_boxed_image_with_the_book_prompt_at_temperature_zero(self):
        client = FakeVisionClient()
        reply = describe_objects("data:image/jpeg;base64,AAAA", "truck", _settings(), client=client)

        self.assertEqual(reply, "Yes, there is one truck in the boxes.")
        request = client.completions.kwargs
        self.assertEqual(request["model"], "gemma3:12b")
        self.assertEqual(request["temperature"], 0.0)
        self.assertIn("truck", request["messages"][0]["content"])
        user_content = request["messages"][1]["content"]
        self.assertIn("one truck or more", user_content[0]["text"])
        self.assertEqual(user_content[1]["image_url"]["url"], "data:image/jpeg;base64,AAAA")


class MultimodalRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        Image.new("RGB", (120, 120), "white").save(self.dir / "00000.png")
        Image.new("RGB", (120, 120), "white").save(self.dir / "00001.png")
        self.samples = [
            Sample(
                id="00000",
                path=self.dir / "00000.png",
                boxes=[[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 20.0, 20.0], [90.0, 10.0, 20.0, 20.0]],
                labels=["car", "car", "truck"],
            ),
            Sample(
                id="00001",
                path=self.dir / "00001.png",
                boxes=[[10.0, 10.0, 30.0, 30.0]],
                labels=["pedestrian"],
            ),
        ]
        self.text_chunks = [
            "Drones identify vehicles with onboard cameras.",
            "Pedestrian detection relies on thermal sensors.",
        ]
        self.llm = FakeLLM()
        self.vision = FakeVisionClient()
        self.rag = MultimodalRAG(
            _settings(),
            self.samples,
            self.text_chunks,
            embedding_function=FakeEmbeddings(),
            llm=self.llm,
            vision_client=self.vision,
            boxed_dir=self.dir / "boxed",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_text_module_answers_from_the_best_chunks(self):
        answer = self.rag.answer("How do drones identify vehicles?")
        self.assertEqual(answer.text_response, "The drone spots trucks among the cars.")
        self.assertEqual(answer.text_context[0], self.text_chunks[0])
        self.assertIn(self.text_chunks[0], self.llm.prompts[0])

    def test_image_module_picks_the_image_and_the_query_names_the_object(self):
        answer = self.rag.answer("How do drones identify a truck?")
        self.assertEqual(answer.sample.id, "00000")
        self.assertEqual(answer.label, "truck")

    def test_vision_module_analyses_a_boxed_copy_of_the_image(self):
        answer = self.rag.answer("How do drones identify a truck?")
        self.assertEqual(answer.boxed_image, self.dir / "boxed" / "boxed_00000.jpg")
        self.assertTrue(answer.boxed_image.is_file())
        self.assertEqual(answer.vision_response, "Yes, there is one truck in the boxes.")
        url = self.vision.completions.kwargs["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    def test_scores_follow_the_book_and_the_modular_score_is_their_mean(self):
        answer = self.rag.answer("How do drones identify a truck?")
        self.assertGreaterEqual(answer.text_score, -1.0)
        self.assertLessEqual(answer.text_score, 1.0)
        self.assertAlmostEqual(
            answer.modular_score, (answer.text_score + answer.vision_score) / 2
        )


class ConfigTests(unittest.TestCase):
    def test_vision_model_defaults_per_provider(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_RAG_VISION_MODEL", None)
            self.assertEqual(Settings.from_env(provider="ollama").vision_model, "gemma3:12b")
            self.assertEqual(Settings.from_env(provider="openai").vision_model, "gpt-4o")

    def test_vision_model_env_var_and_explicit_override(self):
        with mock.patch.dict(os.environ, {"MY_RAG_VISION_MODEL": "llava:7b"}):
            self.assertEqual(Settings.from_env(provider="ollama").vision_model, "llava:7b")
            self.assertEqual(
                Settings.from_env(provider="ollama", vision_model="moondream").vision_model, "moondream"
            )


class CliTests(unittest.TestCase):
    def test_multimodal_rag_and_visdrone_collection_flags_parse(self):
        args = build_parser().parse_args(
            ["--rag", "multimodal", "--show-context", "--vision-model", "gemma3:4b"]
        )
        self.assertEqual(args.rag, "multimodal")
        self.assertTrue(args.show_context)
        self.assertEqual(args.vision_model, "gemma3:4b")

        args = build_parser().parse_args(
            ["--collect", "--corpus", "visdrone", "--limit", "5", "--image-dir", "imgs"]
        )
        self.assertEqual((args.corpus, args.limit, args.image_dir), ("visdrone", 5, "imgs"))

    def test_missing_images_fail_cleanly_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "--rag",
                        "multimodal",
                        "--image-dir",
                        str(Path(tmp) / "no_images"),
                        "--output",
                        str(Path(tmp) / "no_drone.md"),
                        "How do drones identify a truck?",
                    ]
                )
        self.assertEqual(status, 1)
        self.assertIn("--collect --corpus visdrone", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
