"""Exporta o modelo de embeddings de locutor do pyannote (WeSpeakerResNet34)
para ONNX, para rodar na GPU via onnxruntime-directml.

Requer a variavel de ambiente HF_TOKEN com um token de leitura da Hugging Face
que tenha aceitado as condicoes de uso de pyannote/speaker-diarization-3.1.

Uso:
    python gpu_diar/export_embedding.py [--out caminho/para/modelo.onnx]
"""
import argparse
import os

import torch
from pyannote.audio import Pipeline

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wespeaker_resnet34.onnx")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default=DEFAULT_OUT, help=f"caminho do .onnx gerado (padrao: {DEFAULT_OUT})")
args = parser.parse_args()

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    raise SystemExit(
        "Defina HF_TOKEN com um token de leitura da Hugging Face.\n"
        "  Windows: setx HF_TOKEN \"hf_...\"  (abra um terminal novo depois)\n"
        "  Linux/macOS: export HF_TOKEN=\"hf_...\""
    )

OUT_PATH = args.out
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=TOKEN)
model = pipeline._embedding.model_
model.eval()

batch, samples = 4, 32000  # ~2s @16kHz
waveforms = torch.randn(batch, 1, samples)
weights = torch.rand(batch, samples)

with torch.no_grad():
    fbank = model.compute_fbank(waveforms)
    print("fbank shape:", fbank.shape)
    # weights passed to the pooling layer are per-FRAME, not per-sample;
    # ResNet.forward expects weights shaped (batch, frames) matching fbank's time axis.
    frame_weights = torch.rand(batch, fbank.shape[1])
    ref_a, ref_b = model.resnet(fbank, weights=frame_weights)
    print("Reference embed_a/embed_b shapes:", ref_a.shape, ref_b.shape)

torch.onnx.export(
    model.resnet,
    (fbank, frame_weights),
    OUT_PATH,
    input_names=["fbank", "weights"],
    output_names=["embed_a", "embed_b"],
    dynamic_axes={
        "fbank": {0: "batch", 1: "frames"},
        "weights": {0: "batch", 1: "frames"},
        "embed_a": {0: "batch"},
        "embed_b": {0: "batch"},
    },
    opset_version=17,
    dynamo=False,
)
print("Exported ResNet to", OUT_PATH)
