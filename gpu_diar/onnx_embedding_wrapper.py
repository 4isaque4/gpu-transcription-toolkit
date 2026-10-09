"""GPU-accelerated (DirectML) drop-in replacement for pyannote's WeSpeakerResNet34
speaker embedding model. The mel-filterbank feature extraction (compute_fbank) stays
on CPU/PyTorch -- it's cheap and not exportable (uses torch.vmap over a Kaldi-style
fbank routine that traces poorly). Only the heavy ResNet34 forward pass runs via ONNX
Runtime on the GPU.

Usage:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=TOKEN)
    swap_embedding_to_onnx_gpu(pipeline, ONNX_PATH)
    diarization = pipeline(audio_path)
"""
import numpy as np
import onnxruntime as ort
import torch


class OnnxGpuWeSpeakerModel:
    """Mimics the subset of pyannote.audio.core.Model interface that
    PyannoteAudioPretrainedSpeakerEmbedding.__call__ actually uses:
    eval(), to(device) [no-op, GPU work happens inside DirectML], and
    __call__(waveforms, weights=...) -> torch.Tensor embeddings.
    """

    def __init__(self, torch_model, onnx_path, providers=("DmlExecutionProvider", "CPUExecutionProvider")):
        self._torch_model = torch_model  # kept only for compute_fbank + audio/dimension metadata
        self._sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.audio = torch_model.audio
        self.dimension = torch_model.dimension

    def eval(self):
        return self

    def to(self, device):
        # DirectML device selection is fixed at ONNX Runtime session creation time;
        # nothing to move here. Kept for interface compatibility with pyannote.
        return self

    def __call__(self, waveforms: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            fbank = self._torch_model.compute_fbank(waveforms.cpu())
            num_frames = fbank.shape[1]
            if weights is None:
                frame_weights = np.ones((fbank.shape[0], num_frames), dtype=np.float32)
            else:
                w = weights.cpu().numpy().astype(np.float32)
                if w.shape[-1] != num_frames:
                    # weights are per-sample; average-pool down to per-frame to match fbank's time axis
                    frame_weights = _resample_weights_to_frames(w, num_frames)
                else:
                    frame_weights = w

            ort_inputs = {"fbank": fbank.numpy().astype(np.float32), "weights": frame_weights}
            _, embed_b = self._sess.run(["embed_a", "embed_b"], ort_inputs)
            return torch.from_numpy(embed_b)


def _resample_weights_to_frames(weights: np.ndarray, num_frames: int) -> np.ndarray:
    batch, num_samples = weights.shape
    idx = (np.arange(num_frames) * num_samples / num_frames).astype(np.int64)
    idx = np.clip(idx, 0, num_samples - 1)
    return weights[:, idx].astype(np.float32)


def swap_embedding_to_onnx_gpu(pipeline, onnx_path):
    torch_model = pipeline._embedding.model_
    pipeline._embedding.model_ = OnnxGpuWeSpeakerModel(torch_model, onnx_path)
    return pipeline
