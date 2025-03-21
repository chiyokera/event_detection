#!/usr/bin/env python3
import argparse
import io
import os
import wave
from functools import partial
from typing import Literal

import pyopenjtalk
import pysrt
from piper import PiperVoice
from pydub import AudioSegment

VIDEO_DIR = "../../../data/sample/videos"
VOICE_MODEL_DIR = "../../../data/voice_model/en_US-amy-medium.onnx"
CONFIG_FILE = "../../../data/voice_model/en_US-amy-medium.onnx.json"
RESULT_DIR = "../../../data/sample/results"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_name", type=str, required=True)
    parser.add_argument("--lang", type=str, default="en", choices=["ja", "en"])
    return parser.parse_args()


def synthesize_text(voice: PiperVoice, text):
    """
    PiperVoice を使ってテキストから音声合成し、
    合成結果の WAV データを pydub.AudioSegment として返す
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)  # モノラル
        w.setsampwidth(2)  # 16bit (2 bytes)
        w.setframerate(22050)  # サンプルレート（必要に応じて調整）
        voice.synthesize(text, w, length_scale=0.50)
    buf.seek(0)
    segment = AudioSegment.from_file(buf, format="wav")
    return segment


def syntesize_text_openjtalk(text):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)  # モノラル
        w.setsampwidth(2)  # 16bit
        w.setframerate(22050)  # サンプルレート（必要に応じて調整）
        # pyopenjtalk.tts は (wave_data, sample_rate) を返す
        x, sr = pyopenjtalk.tts(text, speed=1.25)
        w.setframerate(sr)  #  (22050のままになるはずだけど、一応)サンプルレート上書き
        w.writeframes(x.astype("int16").tobytes())
    buf.seek(0)
    segment = AudioSegment.from_file(buf, format="wav")
    return segment


def main(args):
    # PiperVoice のモデルを、config_file を指定してロード
    voice = PiperVoice.load(VOICE_MODEL_DIR, CONFIG_FILE)
    # pysrt を用いて SRT ファイルをパース（字幕ブロックを取得）
    input_srt_path = os.path.join(
        RESULT_DIR, args.input_video_name, "srt", "commentary.srt"
    )
    subs = pysrt.open(input_srt_path, encoding="utf-8")
    if len(subs) == 0:
        print("SRT ファイルから字幕が見つかりませんでした。")
        return

    func_syn = synthesize_text
    if args.lang == "ja":
        func_syn = partial(syntesize_text_openjtalk)
        print("Ja mode")
    else:
        func_syn = partial(synthesize_text, voice=voice)
        print("En mode")

    # 出力全体の長さは、最後の字幕ブロックの終了時刻（ミリ秒）とする
    total_duration_ms = subs[-1].end.ordinal
    # 無音の AudioSegment を作成
    output_audio = AudioSegment.silent(duration=total_duration_ms)

    # 各字幕ブロックを順次処理
    for i, sub in enumerate(subs):
        start_ms = sub.start.ordinal
        # 次の字幕ブロックがある場合は、allowed_duration を次の開始時刻との差とする
        if i < len(subs) - 1:
            allowed_duration = subs[i + 1].start.ordinal - start_ms
        else:
            allowed_duration = sub.end.ordinal - start_ms

        # テキストを1行の文字列にまとめる（改行はスペースに置換）
        text = sub.text.replace("\n", " ").strip()
        if not text:
            continue

        # テキストから音声を合成
        synthesized = func_syn(text=text)
        # 合成された音声が許容時間を超える場合はトリミング
        if len(synthesized) > allowed_duration:
            synthesized = synthesized[:allowed_duration]

        # 出力音声の該当部分を置き換える（無音部分を合成音声で上書き）
        output_audio = (
            output_audio[:start_ms]
            + synthesized
            + output_audio[start_ms + len(synthesized) :]
        )

    # 出力WAVファイルにエクスポート
    output_dir = os.path.join(RESULT_DIR, args.input_video_name, "srt_voice")
    os.makedirs(output_dir, exist_ok=True)
    output_wav_path = os.path.join(output_dir, "commentary.wav")
    output_audio.export(output_wav_path, format="wav")
    print(f"音声合成結果を {output_wav_path} に保存しました。")


if __name__ == "__main__":
    import subprocess

    args = parse_arguments()
    main(args)

    video_path = os.path.join(VIDEO_DIR, f"{args.input_video_name}.mp4")
    input_srt_path = os.path.join(
        RESULT_DIR, args.input_video_name, "srt", "commentary.srt"
    )
    video_srt_dir = os.path.join(RESULT_DIR, args.input_video_name, "srt_video")
    os.makedirs(video_srt_dir, exist_ok=True)
    video_srt_path = os.path.join(video_srt_dir, "commentary.mkv")
    cmdline = (
        "ffmpeg -i "
        + video_path
        + " -i "
        + input_srt_path
        + " -map 0:v -map 0:a? -map 1 -metadata:s:s:0 language=en -c:v copy -c:a copy -c:s srt "
        + video_srt_path
    )
    subprocess.call(cmdline, shell=True)

    audio_path = os.path.join(
        RESULT_DIR, args.input_video_name, "srt_voice", "commentary.wav"
    )

    integrated_video_path = os.path.join(video_srt_dir, "integrated_video.mkv")
    cmdline = (
        "ffmpeg -i "
        + video_srt_path
        + " -i "
        + audio_path
        + " -c:v copy -map 0:v:0 -map 1:a:0 -shortest "
        + integrated_video_path
    )
    subprocess.call(cmdline, shell=True)
    # video_srt_pathの削除
    os.remove(video_srt_path)
    print("Finished")
