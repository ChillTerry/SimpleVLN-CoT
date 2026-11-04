import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool"))
import convert_vln
import prepare_sft


class PrepareSFTTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "SFT"
        self.output = self.root / "processed"
        self.episodes = [{"id": 1, "video": "scene/000001", "instructions": ["walk"],
                          "actions": [-1, 1, 2, 3]}]
        annotation = self.data / "annotations/R2R/annotations.json"
        annotation.parent.mkdir(parents=True)
        annotation.write_text(json.dumps(self.episodes))
        rgb = self.data / "R2R/scene/000001/rgb"
        rgb.mkdir(parents=True)
        for index in range(4):
            (rgb / f"{index:03d}.jpg").write_bytes(b"fixture")

    def test_jsonl_preserves_original_samples(self):
        expected = []
        random.seed(0)
        convert_vln.convert_episode_vln(self.episodes[0], str(self.data / "R2R"), expected)
        count = prepare_sft.prepare_dataset(self.data, self.output, "R2R")
        swift = [json.loads(line) for line in (self.output / "ms_swift/R2R.jsonl").read_text().splitlines()]
        self.assertEqual(len(expected), count)
        self.assertEqual(len(swift), count)
        self.assertEqual(sorted(p.relative_to(self.output).as_posix()
                                for p in self.output.rglob("*") if p.is_file()), ["ms_swift/R2R.jsonl"])
        for a, b in zip(expected, swift):
            self.assertEqual([m["role"] for m in b["messages"]], ["system", "user", "assistant"])
            self.assertEqual(b["messages"][0]["content"], a["system"])
            self.assertEqual(b["messages"][1]["content"], a["messages"][0]["value"])
            self.assertEqual(b["messages"][2]["content"], a["messages"][1]["value"])
            self.assertEqual(b["images"], a["images"])

    def test_bad_episode_does_not_publish_partial_datasets(self):
        prepare_sft.prepare_dataset(self.data, self.output, "R2R")
        swift = self.output / "ms_swift/R2R.jsonl"
        original = swift.read_bytes()
        (self.data / "R2R/scene/000001/rgb/001.jpg").unlink()
        with self.assertRaises(UserWarning):
            prepare_sft.prepare_dataset(self.data, self.output, "R2R")
        self.assertEqual(swift.read_bytes(), original)
        self.assertEqual(list(swift.parent.iterdir()), [swift])


if __name__ == "__main__":
    unittest.main()
