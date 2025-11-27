import argparse
import re
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 radio 目录结构复制 main 中的数字命名 wav 到 audio_raw")

    parser.add_argument("--radio-dir", type=str, default="radio",
                        help="原始 radio 子目录名")
    parser.add_argument("--main-dir", type=str, default="main",
                        help="包含源 wav 的 main 子目录名")
    parser.add_argument("--audio-raw-dir", type=str, default="audio_raw",
                        help="输出 audio_raw 子目录名")
    return parser.parse_args()


def main():
    args = parse_args()
    radio_root = Path(args.radio_dir)
    main_root = Path(args.main_dir)
    audio_raw_root = Path(args.audio_raw_dir)
    audio_raw_root.mkdir(parents=True, exist_ok=True)

    digit_wav = re.compile(r"^\d+\.wav$", re.IGNORECASE)
    copied = 0

    for radio_folder in sorted(radio_root.iterdir(), key=lambda p: p.name):
        if not radio_folder.is_dir() or not radio_folder.name.isdigit():
            continue

        src_folder = main_root / radio_folder.name
        if not src_folder.is_dir():
            continue

        dst_folder = audio_raw_root / radio_folder.name
        dst_folder.mkdir(parents=True, exist_ok=True)

        for wav_file in src_folder.glob("*.wav"):
            if not digit_wav.match(wav_file.name):
                continue
            wave_name = Path(wav_file.name)
            target_name = f"{wave_name.stem}_cut_radio_mel{wave_name.suffix}"
            shutil.copy2(wav_file, dst_folder / target_name)
            copied += 1

    print(f"完成复制，共处理 {copied} 个 wav 文件。")


if __name__ == "__main__":
    main()