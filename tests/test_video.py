"""Offline tests for chapter 10 (video stock production: frames + vision + Qdrant).

No OpenCV, video, vision model, OpenAI key, network or Qdrant server is needed:
frame extraction is replaced, the vision client is a fake, the embedding model is
a word-count fake, and the index is a stand-in (plus one real local Qdrant
round-trip, which needs no server).
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from my_rag.cli import build_parser, main
from my_rag.config import (
    DEFAULT_VIDEO_COLLECTION,
    DEFAULT_VIDEO_COMMENTS,
    DEFAULT_VIDEO_DIR,
    DEFAULT_VIDEO_FRAMES,
    DEFAULT_VIDEO_LIMIT,
    DEFAULT_VIDEO_SAMPLE,
    Settings,
)
from my_rag.llm import LLM
from my_rag.scaling import QdrantIndex
from my_rag.video import (
    COMMENT_FILES,
    DEFAULT_THRESHOLD,
    VIDEO_EXPERT_SYSTEM,
    VIDEO_FILES,
    VideoExpert,
    VideoMatch,
    build_chunks,
    classify,
    comment_video,
    describe,
    download_comments,
    download_video,
    duplicate_chunks,
    extract_label,
    extract_rewritten_comment,
    load_comments,
    open_video_index,
    query_video_index,
    read_comments,
    sample_positions,
    save_comments,
    upsert_comments,
    FrameComment,
)

CSV_ROWS = [
    FrameComment("11111111-1111-1111-1111-111111111111", 2, "A basketball player dunks.", "basketball1.mp4"),
    FrameComment("22222222-2222-2222-2222-222222222222", 5, "People jog on a path.", "jogging1.mp4"),
    FrameComment("33333333-3333-3333-3333-333333333333", 8, "A surfer rides a wave.", "surfer1.mp4"),
]

REWRITE_REPLY = (
    "1. No cognitive dissonance.\n"
    "Rewritten Comment:\n"
    "A basketball player leaps and slams the ball through the hoop.\n"
    "\n"
    "Label: basketball"
)


class FakeEmbedding:
    """Word-count vectors keyed to a small vocabulary (deterministic, offline)."""

    vocabulary = ("basketball", "jog", "surfer", "dunk", "wave", "player")

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    def _vector(self, text):
        words = text.lower().split()
        return [float(words.count(term)) + 0.1 for term in self.vocabulary]


class FakeLLM(LLM):
    def __init__(self, reply: str = REWRITE_REPLY) -> None:
        self.reply = reply
        self.calls: list[tuple] = []

    def complete(self, prompt, *, system=None, temperature=None) -> str:
        self.calls.append((prompt, system, temperature))
        return self.reply


class FakePoint:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class FakeIndex:
    """Minimal QdrantIndex stand-in: replays canned matches and records calls."""

    def __init__(self, matches=None, *, results=None, collection: str = DEFAULT_VIDEO_COLLECTION) -> None:
        self.matches = list(matches or [])
        self.results = [list(batch) for batch in results] if results is not None else None
        self.collection = collection
        self.points: list[dict] = []
        self.queries: list[str] = []
        self.closed = False
        self._cursor = 0

    def upsert(self, points, **kwargs):
        self.points.extend(points)

    def query(self, vector, *, limit, with_payload=True):
        self.queries.append(tuple(vector))
        if self.results is not None:
            position = min(self._cursor, len(self.results) - 1)
            self._cursor += 1
            return self.results[position][:limit]
        return self.matches[:limit]

    def count(self):
        return len(self.points)

    def close(self):
        self.closed = True


class FakeVisionCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A person is skiing."))]
        )


class FakeVisionClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeVisionCompletions())


class FakeResponse:
    def __init__(self, content: bytes = b"data", status: int = 200) -> None:
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.calls: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        if any(name in url for name in self.fail):
            return FakeResponse(status=404)
        return FakeResponse()


def _settings(**overrides) -> Settings:
    values = dict(
        provider="ollama",
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        embedding_model="all-minilm",
        embedding_dimension=6,
        vision_model="gemma3:12b",
        video_collection="test_videos",
    )
    values.update(overrides)
    return Settings(**values)


def _write_csv(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "FrameNumber", "Comment", "FileName"])
        for row in rows:
            writer.writerow([row.id, row.frame_number, row.comment, row.file_name])
    return path


class CommentCsvTests(unittest.TestCase):
    def test_save_then_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basketball1.mp4.csv"
            save_comments(CSV_ROWS, path)
            self.assertEqual(read_comments(path), CSV_ROWS)

    def test_read_skips_the_header_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.csv"
            path.write_text(
                "ID,FrameNumber,Comment,FileName\n"
                "1,2,First comment,clip.mp4\n"
                ",,, \n"
                "2,4,,clip.mp4\n"
                "3,6,Second comment,clip.mp4\n",
                encoding="utf-8",
            )
            comments = read_comments(path)
            self.assertEqual([c.comment for c in comments], ["First comment", "Second comment"])
            self.assertEqual(comments[0].frame_number, 2)

    def test_read_missing_file_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            read_comments("/nonexistent/comments.csv")

    def test_load_comments_limits_and_skips_broken_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = _write_csv(Path(tmp) / "a.csv", CSV_ROWS)
            missing = Path(tmp) / "absent.csv"
            progress: list[str] = []
            comments = load_comments([missing, good], limit=2, progress=progress.append)
            self.assertEqual(len(comments), 2)
            self.assertTrue(any("skipped" in line for line in progress))

    def test_load_comments_rejects_an_empty_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_comments([Path(tmp) / "none.csv"])

    def test_build_and_duplicate_chunks(self):
        chunks = build_chunks(CSV_ROWS)
        self.assertEqual(chunks, [c.comment for c in CSV_ROWS])
        self.assertEqual(duplicate_chunks(chunks), 0)
        self.assertEqual(duplicate_chunks(chunks + [chunks[0]]), 1)
        with self.assertRaises(ValueError):
            build_chunks([])


class SamplingTests(unittest.TestCase):
    def test_sample_positions_match_the_book_counter(self):
        # nb=3 over 208 frames prints "Analyzing frame 2, 5, 8, ... 206".
        positions = sample_positions(208, rate=3)
        self.assertEqual(positions[:3], [2, 5, 8])
        self.assertEqual(positions[-1], 206)
        self.assertTrue(all((position + 1) % 3 == 0 and position + 1 < 208 for position in positions))

    def test_sample_positions_rejects_a_bad_rate(self):
        with self.assertRaises(ValueError):
            sample_positions(10, rate=0)

    def test_sample_positions_short_video(self):
        self.assertEqual(sample_positions(2, rate=3), [])


class DownloadTests(unittest.TestCase):
    def test_download_comments_skips_a_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession(fail={"skiing1.mp4.csv"})
            paths = download_comments(Path(tmp), names=("basketball1.mp4.csv", "skiing1.mp4.csv"), session=session)
            self.assertEqual([p.name for p in paths], ["basketball1.mp4.csv"])

    def test_download_video_uses_a_cached_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "alpinist1.mp4"
            cached.write_bytes(b"video")
            session = FakeSession()
            path = download_video("alpinist1.mp4", tmp, session=session)
            self.assertEqual(path, cached)
            self.assertEqual(session.calls, [])  # nothing re-downloaded

    def test_download_video_fetches_from_the_book_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession()
            path = download_video("alpinist1.mp4", tmp, session=session)
            self.assertTrue(path.is_file())
            self.assertIn("Chapter10/videos/alpinist1.mp4", session.calls[0])


class CommentVideoTests(unittest.TestCase):
    def test_comment_video_writes_sampled_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            frame_paths = []
            for number in range(7):
                path = frames_dir / f"frame_{number}.jpg"
                path.write_bytes(b"\xff\xd8\xff")  # a byte blob is enough for the data URL
                frame_paths.append(path)
            output = Path(tmp) / "clip.mp4.csv"
            client = FakeVisionClient()
            with mock.patch("my_rag.video.extract_frames", return_value=frame_paths):
                comments = comment_video(
                    Path(tmp) / "clip.mp4",
                    output,
                    _settings(),
                    frames_dir=frames_dir,
                    sample=3,
                    client=client,
                )
            self.assertEqual([c.frame_number for c in comments], [2, 5])
            self.assertEqual(read_comments(output), comments)
            self.assertEqual(len(client.chat.completions.calls), 2)
            first = client.chat.completions.calls[0]
            self.assertEqual(first["model"], "gemma3:12b")
            self.assertEqual(first["max_tokens"], 150)

    def test_comment_video_limit_caps_the_sampled_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            frame_paths = []
            for number in range(20):
                path = frames_dir / f"frame_{number}.jpg"
                path.write_bytes(b"\xff\xd8\xff")
                frame_paths.append(path)
            client = FakeVisionClient()
            with mock.patch("my_rag.video.extract_frames", return_value=frame_paths):
                comments = comment_video(
                    Path(tmp) / "clip.mp4",
                    Path(tmp) / "clip.csv",
                    _settings(),
                    frames_dir=frames_dir,
                    sample=3,
                    limit=2,
                    client=client,
                )
            self.assertEqual([c.frame_number for c in comments], [2, 5])
            self.assertEqual(len(client.chat.completions.calls), 2)


class IndexTests(unittest.TestCase):
    def test_upsert_comments_stores_text_and_frame_metadata(self):
        index = FakeIndex()
        result = upsert_comments(CSV_ROWS, FakeEmbedding(), index, collection="test_videos")
        self.assertEqual(result.comments, 3)
        self.assertEqual(result.vectors, 3)
        self.assertEqual(result.dimension, 6)
        self.assertEqual(index.points[0]["payload"]["text"], CSV_ROWS[0].comment)
        self.assertEqual(index.points[0]["payload"]["frame_number"], 2)
        self.assertEqual(index.points[0]["payload"]["file_name"], "basketball1.mp4")

    def test_upsert_rejects_an_empty_set(self):
        with self.assertRaises(ValueError):
            upsert_comments([], FakeEmbedding(), FakeIndex())

    def test_query_returns_matches_with_metadata(self):
        point = FakePoint("1", 0.9, {"text": "A dunk.", "frame_number": 4, "file_name": "b.mp4"})
        matches = query_video_index(FakeIndex([point]), FakeEmbedding(), "dunk", top_k=1)
        self.assertEqual(matches, [VideoMatch("1", 0.9, "A dunk.", 4, "b.mp4")])

    def test_open_video_index_round_trips_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(qdrant_path=Path(tmp) / "qdrant")
            index = open_video_index(settings)
            try:
                self.assertIsInstance(index, QdrantIndex)
                self.assertEqual(index.collection, "test_videos")
                upsert_comments(CSV_ROWS, FakeEmbedding(), index, collection=index.collection)
                self.assertEqual(index.count(), 3)
            finally:
                index.close()

    def test_open_video_index_missing_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(qdrant_path=Path(tmp) / "qdrant", video_collection="absent")
            with self.assertRaises(FileNotFoundError):
                open_video_index(settings, create=False)


class ParsingTests(unittest.TestCase):
    def test_extract_rewritten_comment(self):
        self.assertEqual(
            extract_rewritten_comment(REWRITE_REPLY),
            "A basketball player leaps and slams the ball through the hoop.",
        )

    def test_extract_rewritten_comment_without_marker(self):
        self.assertEqual(extract_rewritten_comment("just text"), "")

    def test_extract_rewritten_comment_accepts_alternate_markers_and_strips_markdown(self):
        reply = "**Logical Engaging Style:**\n**A surfer rides a big wave.**\n\n**Label: surfing**"
        self.assertEqual(extract_rewritten_comment(reply), "A surfer rides a big wave.")

    def test_extract_label(self):
        self.assertEqual(extract_label(REWRITE_REPLY), "basketball")
        self.assertEqual(extract_label("no label here"), "")
        self.assertEqual(extract_label("  **Label: surfing**  "), "surfing")


class VideoExpertTests(unittest.TestCase):
    def _expert(self, index=None, llm=None):
        return VideoExpert(
            _settings(),
            index or FakeIndex([FakePoint("1", 0.8, {"text": "A player dunks.", "frame_number": 2, "file_name": "b.mp4"})]),
            embedding_function=FakeEmbedding(),
            llm=llm or FakeLLM(),
        )

    def test_answer_rewrites_the_retrieved_comment(self):
        llm = FakeLLM()
        answer = self._expert(llm=llm).answer("Find a dunk.")
        self.assertEqual(answer.retrieved, "A player dunks.")
        self.assertEqual(answer.rewritten, "A basketball player leaps and slams the ball through the hoop.")
        self.assertEqual(answer.label, "basketball")
        prompt, system, temperature = llm.calls[0]
        self.assertEqual(prompt, "A player dunks.")
        self.assertEqual(system, VIDEO_EXPERT_SYSTEM)
        self.assertEqual(temperature, 0.0)

    def test_evaluate_scores_rewrites_against_the_flashcards(self):
        results = [
            [FakePoint("1", 0.8, {"text": "A basketball player dunks.", "frame_number": 2, "file_name": "b.mp4"})],
            [FakePoint("2", 0.4, {"text": "A surfer rides a wave.", "frame_number": 8, "file_name": "s.mp4"})],
        ]
        expert = self._expert(index=FakeIndex(results=results))
        report = expert.evaluate(
            (("Find a dunk.", "A basketball player slams the ball."), ("Find a wave.", "A surfer on a wave."))
        )
        self.assertEqual(len(report.items), 2)
        self.assertEqual(report.retrieval_scores, [0.8, 0.4])
        self.assertGreater(report.items[0].similarity, 0)
        self.assertIsInstance(report.descriptive, type(describe([1.0])))
        self.assertEqual(report.classification.threshold, DEFAULT_THRESHOLD)


class MetricsTests(unittest.TestCase):
    def test_describe(self):
        stats = describe([0.2, 0.4, 0.6, 0.8])
        self.assertAlmostEqual(stats.mean, 0.5)
        self.assertAlmostEqual(stats.median, 0.5)
        self.assertAlmostEqual(stats.minimum, 0.2)
        self.assertAlmostEqual(stats.maximum, 0.8)
        self.assertAlmostEqual(stats.range, 0.6)
        self.assertAlmostEqual(stats.q1, 0.35)
        self.assertAlmostEqual(stats.q3, 0.65)
        self.assertAlmostEqual(stats.iqr, 0.3)

    def test_describe_rejects_empty(self):
        with self.assertRaises(ValueError):
            describe([])

    def test_classify_perfect_and_confusion_matrix(self):
        report = classify([0.9, 0.9], [0.9, 0.9], threshold=0.6)
        self.assertEqual(report.f1, 1.0)
        self.assertEqual(report.precision, 1.0)
        self.assertEqual(report.recall, 1.0)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.confusion, [[0, 0], [0, 2]])

    def test_classify_zero_guards_and_mismatch(self):
        report = classify([0.1, 0.1], [0.9, 0.9], threshold=0.6)
        self.assertEqual(report.confusion, [[0, 0], [2, 0]])
        self.assertEqual(report.recall, 0.0)
        self.assertEqual(report.f1, 0.0)
        with self.assertRaises(ValueError):
            classify([0.1], [0.1, 0.2], threshold=0.6)


class ConfigTests(unittest.TestCase):
    def test_chapter_10_defaults_are_registered(self):
        self.assertEqual(DEFAULT_VIDEO_COLLECTION, "videos-sports-us")
        self.assertEqual(DEFAULT_VIDEO_SAMPLE, 3)
        self.assertEqual(DEFAULT_VIDEO_LIMIT, 6)
        self.assertEqual(DEFAULT_VIDEO_DIR, Path("data/raw/videos"))
        self.assertEqual(DEFAULT_VIDEO_COMMENTS, Path("data/raw/video_comments"))
        self.assertEqual(DEFAULT_VIDEO_FRAMES, Path("data/raw/video_frames"))
        self.assertEqual(len(VIDEO_FILES), 21)
        self.assertEqual(len(COMMENT_FILES), 21)
        self.assertTrue(all(name.endswith(".mp4.csv") for name in COMMENT_FILES))

    def test_settings_read_the_video_env(self):
        with mock.patch.dict(
            "os.environ",
            {
                "MY_RAG_VIDEO_DIR": "/tmp/videos",
                "MY_RAG_VIDEO_COMMENTS": "/tmp/comments",
                "MY_RAG_VIDEO_COLLECTION": "custom-videos",
                "MY_RAG_VIDEO_SAMPLE": "5",
                "MY_RAG_VIDEO_LIMIT": "3",
            },
            clear=False,
        ):
            settings = Settings.from_env(provider="ollama")
        self.assertEqual(settings.video_dir, Path("/tmp/videos"))
        self.assertEqual(settings.video_comments, Path("/tmp/comments"))
        self.assertEqual(settings.video_collection, "custom-videos")
        self.assertEqual(settings.video_sample, 5)
        self.assertEqual(settings.video_limit, 3)


class CliTests(unittest.TestCase):
    def test_chapter_10_flags_parse(self):
        args = build_parser().parse_args(["--comment-video", "--video", "alpinist1.mp4", "--sample", "4"])
        self.assertEqual((args.comment_video, args.video, args.sample), (True, "alpinist1.mp4", 4))
        args = build_parser().parse_args(["--build-video-index", "--limit", "20"])
        self.assertEqual((args.build_video_index, args.limit), (True, 20))
        args = build_parser().parse_args(["--rag", "video", "--evaluate"])
        self.assertEqual((args.rag, args.evaluate), ("video", True))

    def test_collect_video_comments_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with mock.patch(
                "my_rag.video.download_comments", return_value=[Path(tmp) / "a.csv"]
            ) as fetch:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                    status = main(["--collect", "--corpus", "video-comments", "--comments-dir", tmp])
        self.assertEqual(status, 0)
        fetch.assert_called_once()
        self.assertIn("wrote 1 comment files", stderr.getvalue())

    def test_build_video_index_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(Path(tmp) / "basketball1.mp4.csv", CSV_ROWS)
            index = FakeIndex()
            with mock.patch("my_rag.video.open_video_index", return_value=index), mock.patch(
                "my_rag.cli.create_embedding_function", return_value=FakeEmbedding()
            ):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = main(["--build-video-index", "--comments-dir", tmp, "--collection", "videos-sports-us"])
        self.assertEqual(status, 0)
        self.assertEqual(index.count(), 3)
        self.assertTrue(index.closed)

    def test_build_video_index_without_comments_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                status = main(["--build-video-index", "--comments-dir", tmp])
        self.assertEqual(status, 1)
        self.assertIn("error:", stderr.getvalue())

    def test_video_query_offline(self):
        point = FakePoint("1", 0.8, {"text": "A basketball player dunks.", "frame_number": 2, "file_name": "b.mp4"})
        index = FakeIndex([point])
        stdout = io.StringIO()
        with mock.patch("my_rag.video.open_video_index", return_value=index), mock.patch(
            "my_rag.video.create_embedding_function", return_value=FakeEmbedding()
        ), mock.patch("my_rag.video.create_llm", return_value=FakeLLM()):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                status = main(["--rag", "video", "--show-context", "Find a dunk."])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("A basketball player dunks.", out)
        self.assertIn("Label: basketball", out)
        self.assertTrue(index.closed)

    def test_video_query_evaluate_offline(self):
        results = [
            [FakePoint("1", 0.8, {"text": "A basketball player dunks.", "frame_number": 2, "file_name": "b.mp4"})],
        ]
        index = FakeIndex(results=results)
        stdout = io.StringIO()
        with mock.patch("my_rag.video.open_video_index", return_value=index), mock.patch(
            "my_rag.video.create_embedding_function", return_value=FakeEmbedding()
        ), mock.patch("my_rag.video.create_llm", return_value=FakeLLM()):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                status = main(["--rag", "video", "--evaluate", "--limit", "1"])
        self.assertEqual(status, 0)
        out = stdout.getvalue()
        self.assertIn("Mean:", out)
        self.assertIn("F1 Score:", out)
        self.assertIn("Confusion matrix:", out)

    def test_comment_video_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"video")
            frames = Path(tmp) / "frames"
            frames.mkdir()
            paths = []
            for number in range(4):
                path = frames / f"frame_{number}.jpg"
                path.write_bytes(b"\xff\xd8\xff")
                paths.append(path)
            with mock.patch("my_rag.video.extract_frames", return_value=paths), mock.patch(
                "my_rag.video.comment_frame", return_value="A caption."
            ):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = main(
                        ["--comment-video", "--video", str(video), "--frames-dir", str(frames),
                         "--comments-dir", tmp, "--sample", "3"]
                    )
            self.assertEqual(status, 0)
            self.assertTrue((Path(tmp) / "clip.mp4.csv").is_file())

    def test_video_info_without_opencv_reports_a_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"video")
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                status = main(["--video-info", "--video", str(video)])
        # Without OpenCV the command must fail gracefully (exit 1 with an error), not traceback.
        self.assertIn(status, (0, 1))
        if status == 1:
            self.assertIn("error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
