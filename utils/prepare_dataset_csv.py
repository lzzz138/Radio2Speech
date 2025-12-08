import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 train/val/test CSV 列表")
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="包含 radio/ 与 audio/ 子目录的数据集根目录")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="CSV 输出目录")
    parser.add_argument("--radio-dir-name", default="radio", type=str,
                        help="射频子目录名称")
    parser.add_argument("--audio-dir-name", default="audio", type=str,
                        help="语音子目录名称")
    return parser.parse_args()


def infer_split(folder_idx: int) -> str:
    # if 0 <= folder_idx <= 4:
    #     return "test"
    if 0 <= folder_idx <= 9:
        return "val"
    return "train"


def get_mel_length(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as handle:
        if "mel" in handle:
            return handle["mel"].shape[0]
        if "feats" in handle:
            return handle["feats"].shape[0]
        raise KeyError(f"{h5_path} 不包含 'mel' 或 'feats' 数据集")


def collect_pairs(dataset_root: Path,
                  radio_dir_name: str,
                  audio_dir_name: str) -> Dict[str, List[Tuple[str, str, int]]]:
    radio_root = dataset_root / radio_dir_name
    audio_root = dataset_root / audio_dir_name
    if not radio_root.is_dir() or not audio_root.is_dir():
        raise FileNotFoundError("radio 或 audio 子目录不存在")

    splits: Dict[str, List[Tuple[str, str, int]]] = {"train": [], "val": [], "test": []}

    for audio_folder in sorted(audio_root.iterdir(), key=lambda p: p.name):
        if not audio_folder.is_dir():
            continue
        folder_idx = int(audio_folder.name)
        split = infer_split(folder_idx)

        radio_folder = radio_root / audio_folder.name
        if not radio_folder.is_dir():
            continue

        for audio_file in sorted(audio_folder.glob("*.h5")):
            radio_file = radio_folder / audio_file.name.replace("audio", "radio")
            if not radio_file.is_file():
                continue
            mel_len = get_mel_length(audio_file)
            splits[split].append((
                str(radio_file.resolve()),
                str(audio_file.resolve()),
                mel_len,
            ))
    return splits


def write_csv(output_dir: Path,
              splits: Dict[str, List[Tuple[str, str, int]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        csv_path = output_dir / f"{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"{split}: 写入 {len(rows)} 条 -> {csv_path}")


def main() -> None:
    args = parse_args()
    splits = collect_pairs(args.dataset_root,
                           args.radio_dir_name,
                           args.audio_dir_name)
    write_csv(args.output_dir, splits)


if __name__ == "__main__":
    main()