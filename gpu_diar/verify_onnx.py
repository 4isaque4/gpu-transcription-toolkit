"""Valida numericamente o modelo ONNX de embeddings (GPU/DirectML) contra a
referencia em PyTorch (CPU), antes de confiar nele em producao.

Requer a variavel de ambiente HF_TOKEN com um token de leitura da Hugging Face.

Uso:
    python gpu_diar/verify_onnx.py [--onnx caminho/para/modelo.onnx]

Espera-se `Max abs diff: 0.000000` entre as duas implementacoes.
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from pyannote.audio import Pipeline

DEFAULT_ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wespeaker_resnet34.onnx")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--onnx", default=DEFAULT_ONNX, help=f"caminho do .onnx a validar (padrao: {DEFAULT_ONNX})")
args = parser.parse_args()

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    raise SystemExit(
        "Defina HF_TOKEN com um token de leitura da Hugging Face.\n"
        "  Windows: setx HF_TOKEN \"hf_...\"  (abra um terminal novo depois)\n"
        "  Linux/macOS: export HF_TOKEN=\"hf_...\""
    )

ONNX_PATH = args.onnx
if not os.path.exists(ONNX_PATH):
    raise SystemExit(f"Modelo nao encontrado em {ONNX_PATH}. Rode gpu_diar/export_embedding.py antes.")

print("Available ORT providers:", ort.get_available_providers())

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=TOKEN)
model = pipeline._embedding.model_
model.eval()

torch.manual_seed(0)
batch, samples = 8, 48000  # 3s @16kHz, different from export-time trace shape
waveforms = torch.randn(batch, 1, samples)

with torch.no_grad():
    fbank = model.compute_fbank(waveforms)
    frame_weights = torch.rand(batch, fbank.shape[1])
    t0 = time.time()
    _, torch_embed = model.resnet(fbank, weights=frame_weights)
    torch_time = time.time() - t0

sess = ort.InferenceSession(ONNX_PATH, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
print("Session providers in use:", sess.get_providers())

ort_inputs = {"fbank": fbank.numpy(), "weights": frame_weights.numpy()}
t0 = time.time()
_, onnx_embed = sess.run(["embed_a", "embed_b"], ort_inputs)
onnx_time = time.time() - t0

torch_embed_np = torch_embed.numpy()
diff = np.abs(torch_embed_np - onnx_embed)
print(f"PyTorch(CPU) time: {torch_time*1000:.1f} ms | ONNX(DirectML) time: {onnx_time*1000:.1f} ms")
print(f"Max abs diff: {diff.max():.6f} | Mean abs diff: {diff.mean():.6f}")
print(f"Torch embed norm sample: {np.linalg.norm(torch_embed_np[0]):.4f} | ONNX embed norm sample: {np.linalg.norm(onnx_embed[0]):.4f}")

# cosine similarity between corresponding embeddings, sanity check they represent "the same" vector
cos = (torch_embed_np * onnx_embed).sum(axis=1) / (
    np.linalg.norm(torch_embed_np, axis=1) * np.linalg.norm(onnx_embed, axis=1)
)
print("Per-sample cosine similarity (torch vs onnx):", cos)
