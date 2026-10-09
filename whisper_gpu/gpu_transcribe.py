import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


TIMESTAMP_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2}\.\d{3})\s+-->\s+"
    r"(\d{2}):(\d{2}):(\d{2}\.\d{3})\]"
)


def atomic_json(path: Path, payload) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def write_status(path: Path | None, stage: str, pct: float, detail: str) -> None:
    if path is None:
        return
    atomic_json(
        path,
        {
            "stage": stage,
            "pct": round(pct, 1),
            "detail": detail,
            "updated_at": time.strftime("%H:%M:%S"),
        },
    )


def log(handle, message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def timestamp_seconds(groups) -> float:
    hours, minutes, seconds = groups
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def is_control_token(text: str) -> bool:
    return text.startswith("[_") and text.endswith("]")


def convert_whisper_json(source: Path, destination: Path) -> int:
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    converted = []
    for segment in raw.get("transcription", []):
        words = []
        for token in segment.get("tokens", []):
            token_text = token.get("text", "")
            offsets = token.get("offsets") or {}
            if not token_text or is_control_token(token_text):
                continue

            start = float(offsets.get("from", 0)) / 1000.0
            end = float(offsets.get("to", offsets.get("from", 0))) / 1000.0
            begins_word = token_text[:1].isspace()

            if not words or begins_word:
                words.append({"start": start, "end": end, "word": token_text})
            else:
                words[-1]["word"] += token_text
                words[-1]["end"] = max(words[-1]["end"], end)

        offsets = segment.get("offsets") or {}
        converted.append(
            {
                "start": float(offsets.get("from", 0)) / 1000.0,
                "end": float(offsets.get("to", offsets.get("from", 0))) / 1000.0,
                "text": segment.get("text", ""),
                "words": words,
            }
        )

    atomic_json(destination, converted)
    return len(converted)


def run_checked(command, log_handle) -> None:
    completed = subprocess.run(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Comando falhou com codigo {completed.returncode}: {command[0]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--whisper", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vad-model", type=Path)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--no-status", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    status_path = None if args.no_status else args.out_dir / "status.json"
    transcript_path = args.out_dir / "transcript_words.json"
    full_json_base = args.work_dir / "large-v3-gpu"
    full_json_path = full_json_base.with_suffix(".json")
    wav_path = args.work_dir / "audio-16khz-mono.wav"
    log_path = args.out_dir / "gpu_transcription.log"

    with log_path.open("a", encoding="utf-8") as log_handle:
        try:
            log(log_handle, "=== INICIO DA TRANSCRICAO GPU/VULKAN ===")
            if not wav_path.exists():
                write_status(status_path, "preparando_gpu", 50.0, "convertendo audio para WAV")
                log(log_handle, "Convertendo audio para WAV mono 16 kHz...")
                run_checked(
                    [
                        str(args.ffmpeg),
                        "-hide_banner",
                        "-loglevel",
                        "warning",
                        "-y",
                        "-i",
                        str(args.audio),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        str(wav_path),
                    ],
                    log_handle,
                )

            write_status(status_path, "transcricao_gpu", 50.0, "carregando large-v3 na GPU")
            command = [
                str(args.whisper),
                "-m",
                str(args.model),
                "-f",
                str(wav_path),
                "-l",
                "pt",
                "-bs",
                "5",
                "-bo",
                "5",
                "-t",
                "4",
                "-ojf",
                "-of",
                str(full_json_base),
            ]
            if args.vad_model:
                command.extend(
                    [
                        "-mc",
                        "0",
                        "-sns",
                        "--vad",
                        "-vm",
                        str(args.vad_model),
                        "-vsd",
                        "500",
                        "-vmsd",
                        "30",
                        "-vp",
                        "200",
                    ]
                )
            log(log_handle, "Executando Whisper large-v3 na GPU AMD via Vulkan...")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for output_line in process.stdout:
                sys.stdout.write(output_line)
                log_handle.write(output_line)
                log_handle.flush()
                match = TIMESTAMP_RE.search(output_line)
                if match:
                    end_seconds = timestamp_seconds(match.groups()[3:])
                    audio_pct = min(100.0, 100.0 * end_seconds / args.duration)
                    write_status(
                        status_path,
                        "transcricao_gpu",
                        50.0 + 0.5 * audio_pct,
                        f"audio processado: {end_seconds:.0f}s/{args.duration:.0f}s ({audio_pct:.0f}%)",
                    )
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"whisper-cli terminou com codigo {return_code}")
            if not full_json_path.exists():
                raise RuntimeError("whisper-cli nao produziu o JSON esperado")

            write_status(status_path, "finalizando_gpu", 99.5, "convertendo timestamps")
            segment_count = convert_whisper_json(full_json_path, transcript_path)
            log(log_handle, f"Transcricao concluida: {segment_count} segmentos.")
            write_status(
                status_path,
                "transcricao",
                100.0,
                f"concluida na GPU - {segment_count} segmentos",
            )
            log(log_handle, "=== TRANSCRICAO GPU COMPLETA ===")
            return 0
        except Exception as exc:
            log(log_handle, f"ERRO: {exc}")
            write_status(status_path, "erro_gpu", 50.0, str(exc))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
