"""Convert the extracted SimpleVLN SFT release into ms-swift JSONL files."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import random
import tempfile
import warnings

from tqdm import tqdm

import convert_common as common
import convert_vln


DATASETS = ("R2R", "RxR", "ScaleVLN")


def prepare_dataset(data_root, output_root, name, seed=0, max_history=8):
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    if name not in DATASETS or max_history < 1:
        raise ValueError("Choose a released dataset and a positive history count")
    annotation = data_root / "annotations" / name / "annotations.json"
    image_root = data_root / name
    if not annotation.is_file() or not image_root.is_dir():
        raise ValueError(f"Missing annotations or images for {name} under {data_root}")
    swift_path = output_root / "ms_swift" / f"{name}.jsonl"
    swift_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = common.load_episodes(annotation)
    random.seed(seed)
    sample_count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=swift_path.parent,
            prefix=f".{swift_path.name}.", suffix=".partial", delete=False,
        ) as swift:
            temporary = Path(swift.name)
            for episode in tqdm(episodes, desc=f"prepare {name}", mininterval=10):
                samples = []
                with warnings.catch_warnings():
                    # Stop on invalid trajectories rather than silently skipping them.
                    warnings.simplefilter("error", UserWarning)
                    convert_vln.convert_episode_vln(
                        episode, str(image_root), samples, max_history=max_history,
                    )
                if not samples:
                    raise ValueError(f"No samples for {name} episode {common.get_episode_id(episode)}")
                original_images = set()
                for sample in samples:
                    original_images.update(sample["images"])
                for image in original_images:
                    path = Path(image)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"Missing or empty image: {path}")
                for sample in samples:
                    user, assistant = sample["messages"]
                    images = sample["images"]
                    if (user["from"] != "user" or assistant["from"] != "assistant"
                            or user["value"].count("<image>") != len(images)
                            or not 2 <= len(images) <= max_history + 1
                            or not assistant["value"].strip()):
                        raise ValueError(f"Invalid multimodal sample: {sample['episode_id']}")
                    labels = assistant["value"].split(", ")
                    for label in labels:
                        common.to_action_text(label)
                    if "stop" in labels and labels[-1] != "stop":
                        raise ValueError("STOP must be the final action")
                    swift_sample = {
                        "messages": [
                            {"role": "system", "content": sample["system"]},
                            {"role": "user", "content": user["value"]},
                            {"role": "assistant", "content": assistant["value"]},
                        ],
                        "images": list(images),
                        "episode_id": sample["episode_id"],
                        "dataset": name,
                    }
                    swift.write(json.dumps(swift_sample, ensure_ascii=False) + "\n")
                    sample_count += 1
        if not sample_count:
            raise ValueError(f"No samples produced for {name}")
        os.replace(temporary, swift_path)
        print(f"PREPARED {name}: {len(episodes)} trajectories -> {sample_count} samples", flush=True)
        return sample_count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Extracted SFT directory containing R2R, RxR, ScaleVLN and annotations")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_history", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3,
                        help="Datasets to prepare concurrently (default: 3)")
    args = parser.parse_args()
    if args.max_history < 1 or args.workers < 1:
        parser.error("--max_history and --workers must be positive")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.datasets))) as pool:
        futures = [pool.submit(prepare_dataset, args.data_root, args.output_root,
                               name, args.seed, args.max_history) for name in args.datasets]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
