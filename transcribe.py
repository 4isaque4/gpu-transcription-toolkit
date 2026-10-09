"""Pipeline completo: diarizacao (pyannote, GPU via ONNX/DirectML quando o modelo
exportado existe) seguida de transcricao.

Se ja houver transcript_words.json no diretorio de saida - por exemplo gerado pelo
whisper_gpu/gpu_transcribe.py - a transcricao e pulada e o arquivo e reusado.
Caso contrario, transcreve com faster-whisper em CPU.

Requer a variavel de ambiente HF_TOKEN com um token de leitura da Hugging Face.

Uso:
    python transcribe.py --audio caminho/do/audio.mp3 --out-dir pasta/de/saida
"""
import argparse
import os
import sys
import json
import time

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ONNX = os.path.join(_REPO_DIR, "gpu_diar", "wespeaker_resnet34.onnx")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--audio", required=True, help="arquivo de audio a processar")
parser.add_argument("--out-dir", required=True, help="pasta onde gravar JSONs, progresso e status")
parser.add_argument("--onnx", default=_DEFAULT_ONNX,
                    help="modelo ONNX de embeddings; se nao existir, a diarizacao roda em CPU")
parser.add_argument("--duration", type=float, default=None,
                    help="duracao do audio em segundos; so e usada se a deteccao automatica falhar")
args = parser.parse_args()

AUDIO_PATH = args.audio
OUT_DIR = args.out_dir
ONNX_PATH = args.onnx
AUDIO_DURATION_S = args.duration

if not os.path.exists(AUDIO_PATH):
    raise SystemExit(f"Audio nao encontrado: {AUDIO_PATH}")
os.makedirs(OUT_DIR, exist_ok=True)

DIAR_JSON = os.path.join(OUT_DIR, "diarization.json")
TRANSCRIPT_JSON = os.path.join(OUT_DIR, "transcript_words.json")
PROGRESS_FILE = os.path.join(OUT_DIR, "progress.txt")
STATUS_FILE = os.path.join(OUT_DIR, "status.json")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_status(stage, pct, detail=""):
    # Status reporting must NEVER crash the pipeline (e.g. Windows file-lock
    # contention from the live dashboard reading status.json concurrently).
    payload = {
        "stage": stage,
        "pct": round(pct, 1),
        "detail": detail,
        "updated_at": time.strftime("%H:%M:%S"),
    }
    try:
        tmp = STATUS_FILE + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        for attempt in range(8):
            try:
                os.replace(tmp, STATUS_FILE)
                break
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(0.1)
    except Exception as e:
        print(f"[status write skipped: {e}]", flush=True)

class DiarizationProgressHook:
    """Reports per-step progress from pyannote into progress.txt / status.json.
    Diarization is treated as 0-50% of the overall pipeline; transcription as 50-100%.
    """
    def __init__(self):
        self._last_step = None
        self._last_logged_pct = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None or total is None or total == 0:
            completed, total = 1, 1
        step_pct = 100.0 * completed / total
        overall_pct = 0.0 + 0.5 * step_pct
        write_status("diarizacao", overall_pct, f"{step_name}: {completed}/{total}")
        if step_name != self._last_step:
            self._last_step = step_name
            self._last_logged_pct = -1
            log(f"Diarizacao - etapa '{step_name}' iniciada")
        if step_pct - self._last_logged_pct >= 10 or completed >= total:
            self._last_logged_pct = step_pct
            log(f"  Diarizacao - '{step_name}': {completed}/{total} ({step_pct:.0f}%) | geral ~{overall_pct:.0f}%")

def step_diarization():
    if os.path.exists(DIAR_JSON):
        log("Diarizacao ja existe, pulando etapa.")
        write_status("diarizacao", 50.0, "ja concluida (cache)")
        return
    log("Iniciando diarizacao (pyannote)...")
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)

    if os.path.exists(ONNX_PATH):
        sys.path.insert(0, os.path.dirname(os.path.abspath(ONNX_PATH)))
        from onnx_embedding_wrapper import swap_embedding_to_onnx_gpu
        swap_embedding_to_onnx_gpu(pipeline, ONNX_PATH)
        log("Embeddings de voz acelerados via GPU (ONNX+DirectML).")
    else:
        log("Modelo ONNX de embeddings nao encontrado, usando CPU/PyTorch (mais lento).")

    log("Pipeline de diarizacao carregado. Processando audio...")
    with DiarizationProgressHook() as hook:
        diarization = pipeline(AUDIO_PATH, hook=hook)
    segments = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    with open(DIAR_JSON, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    n_speakers = len(set(s["speaker"] for s in segments))
    log(f"Diarizacao concluida: {len(segments)} segmentos de fala, {n_speakers} falantes detectados.")
    write_status("diarizacao", 50.0, f"concluida - {n_speakers} falantes")

def step_transcription():
    if os.path.exists(TRANSCRIPT_JSON):
        log("Transcricao ja existe, pulando etapa.")
        write_status("transcricao", 100.0, "ja concluida (cache)")
        return
    log("Iniciando transcricao com faster-whisper (large-v3)...")
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=16)
    log("Modelo large-v3 carregado. Transcrevendo audio completo...")
    segments_gen, info = model.transcribe(
        AUDIO_PATH,
        language="pt",
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        best_of=5,
    )
    total_duration = info.duration or AUDIO_DURATION_S
    if not total_duration:
        raise SystemExit("Nao foi possivel detectar a duracao do audio; passe --duration em segundos.")
    log(f"Idioma detectado/forcado: {info.language} (prob {info.language_probability:.2f}). Duracao: {total_duration:.1f}s")
    words = []
    seg_count = 0
    last_logged_pct = -1
    for seg in segments_gen:
        seg_count += 1
        seg_words = []
        if seg.words:
            for w in seg.words:
                seg_words.append({"start": w.start, "end": w.end, "word": w.word})
        words.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": seg_words,
        })
        audio_pct = 100.0 * seg.end / total_duration
        overall_pct = 50.0 + 0.5 * audio_pct
        write_status("transcricao", overall_pct, f"audio processado: {seg.end:.0f}s/{total_duration:.0f}s ({audio_pct:.0f}%)")
        if audio_pct - last_logged_pct >= 5:
            last_logged_pct = audio_pct
            log(f"  Transcricao: {seg_count} segmentos | {seg.end:.0f}s/{total_duration:.0f}s ({audio_pct:.0f}%) | geral ~{overall_pct:.0f}%")
        with open(TRANSCRIPT_JSON, "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
    log(f"Transcricao concluida: {seg_count} segmentos.")
    write_status("transcricao", 100.0, f"concluida - {seg_count} segmentos")

if __name__ == "__main__":
    start = time.time()
    log("=== INICIO DO PROCESSAMENTO ===")
    write_status("iniciando", 0.0, "")
    step_diarization()
    step_transcription()
    elapsed = time.time() - start
    log(f"=== PROCESSAMENTO COMPLETO em {elapsed/60:.1f} minutos ===")
    write_status("completo", 100.0, f"{elapsed/60:.1f} min totais")
