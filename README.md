# GPU-accelerated transcription + diarization (AMD RX 6700 XT / Windows)

Setup usado para transcrever e diarizar (separar falantes) áudios longos em português,
acelerado pela GPU AMD desta máquina (Vulkan para o Whisper, DirectML para os embeddings
de locutor do pyannote). Todo o restante (pyannote segmentation, VAD, conversão de áudio)
continua em CPU — só as partes pesadas foram movidas para a GPU.

Hardware detectado: **AMD Radeon RX 6700 XT** (driver Vulkan/DirectX12 nativo do Windows).

Resultado medido nesta máquina, áudio de ~93 minutos:
- Transcrição (whisper.cpp + Vulkan, large-v3): **~8,4 min** de processamento total.
- Diarização (pyannote + ONNX/DirectML para os embeddings): **~4 min**, contra ~50+ min
  na CPU (segmentação continua em CPU, é rápida; a etapa de embeddings, que é o gargalo,
  passou a rodar na GPU com ganho de **~11,5x**, saída numericamente idêntica à da CPU).

## Estrutura

```
whisper_gpu/           transcrição com whisper.cpp compilado com backend Vulkan
  gpu_transcribe.py     converte audio -> wav 16kHz mono, roda whisper-cli, converte JSON
  handoff_to_gpu.ps1     script de orquestração de exemplo (dispara a transcrição GPU)

gpu_diar/               diarização (pyannote) com embeddings acelerados via ONNX+DirectML
  export_embedding.py    exporta o modelo WeSpeakerResNet34 (embeddings) para ONNX
  onnx_embedding_wrapper.py  substitui o modelo PyTorch do pyannote por uma sessão ONNX/DirectML
  verify_onnx.py         valida numericamente ONNX(GPU) vs PyTorch(CPU) antes de usar em produção

transcribe.py            pipeline completo: diarização (GPU se o .onnx existir) + transcrição
                          via faster-whisper em CPU como fallback, ou pulando se já houver
                          transcript_words.json (ex.: gerado pelo whisper.cpp/GPU)

merge_transcript.py      combina diarização + transcrição num Markdown final, por falante
```

O `gpu_diar/wespeaker_resnet34.onnx` **não é versionado aqui**: tem 26 MB e é derivado de um
modelo *gated* na Hugging Face, cuja licença não permite redistribuição avulsa. Gere o seu
com `gpu_diar/export_embedding.py` — leva menos de um minuto e só precisa ser refeito se o
checkpoint `pyannote/speaker-diarization-3.1` for atualizado.

## Antes de começar: token da Hugging Face

Os modelos do pyannote são *gated*. É preciso ter conta na Hugging Face, aceitar as condições
de uso de `pyannote/speaker-diarization-3.1` e gerar um token de **leitura**. Todos os scripts
leem esse token da variável de ambiente `HF_TOKEN`; nunca escreva o valor dentro do código.

```bash
# Windows - abra um terminal novo depois, a variável só existe em processos iniciados dali
setx HF_TOKEN "hf_..."

# Linux / macOS
export HF_TOKEN="hf_..."
```

## Por que dois caminhos de GPU diferentes

- **Transcrição (Whisper):** o `faster-whisper` (CTranslate2) só acelera em CUDA (Nvidia) —
  não há suporte a AMD. A solução foi usar o **whisper.cpp**, que tem backend Vulkan
  multiplataforma e funciona nativamente com a RX 6700 XT.
- **Diarização (pyannote):** é PyTorch puro. `torch-directml` é limitado (trava em
  PyTorch <= 2.3.1 e não cobre todas as operações). A solução foi **exportar só o modelo
  de embeddings de locutor para ONNX** e rodar via `onnxruntime-directml`, que aceita
  qualquer GPU DirectX12. A segmentação (rápida) e o resto da lógica do pipeline
  continuam em PyTorch/CPU sem necessidade de mudança.

## Setup: transcrição via whisper.cpp + Vulkan

1. **Vulkan SDK** instalado (necessário para compilar o backend Vulkan do whisper.cpp).
2. Clonar o whisper.cpp em um caminho **curto** (ex.: `C:\wcpp`) — caminhos longos
   quebram a compilação no Windows (limite clássico de 260 caracteres):
   ```bash
   git clone https://github.com/ggml-org/whisper.cpp.git C:\wcpp
   ```
3. Compilar com Vulkan habilitado (usar `-B` com pasta curta também, ex. `C:\wcpp\build-vulkan`):
   ```bash
   cmake -B C:\wcpp\build-vulkan -S C:\wcpp -DGGML_VULKAN=ON
   cmake --build C:\wcpp\build-vulkan --config Release
   ```
   Binário final (nesta máquina): `C:\wb\bin\Release\whisper-cli.exe` (copiado do build).
   Ao iniciar, o log deve mostrar algo como:
   `ggml_vulkan: 0 = AMD Radeon RX 6700 XT (AMD proprietary driver) | fp16: 1`.
4. Baixar os modelos GGML (não versionados aqui por serem grandes):
   - `ggml-large-v3.bin` (~3 GB) — modelo Whisper.
   - `ggml-silero-v6.2.0.bin` — VAD (evita alucinação/repetição em silêncios longos,
     ver nota abaixo).
   ```bash
   curl -L -o ggml-large-v3.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
   curl -L -o ggml-silero-v6.2.0.bin https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
   ```
5. Rodar:
   ```bash
   python whisper_gpu/gpu_transcribe.py \
     --audio caminho\para\audio.mp3 \
     --out-dir pasta_de_saida \
     --work-dir pasta_de_trabalho \
     --ffmpeg caminho\para\ffmpeg.exe \
     --whisper C:\wb\bin\Release\whisper-cli.exe \
     --model C:\wmodels\ggml-large-v3.bin \
     --vad-model C:\wmodels\ggml-silero-v6.2.0.bin \
     --duration <duracao_do_audio_em_segundos>
   ```
   Gera `transcript_words.json` (mesmo formato usado por `transcribe.py`) em `--out-dir`.

   **Importante (lição aprendida):** sem `--vad-model`, o whisper.cpp pode "alucinar"
   e repetir a última frase em loop durante silêncios longos. Os parâmetros de VAD
   validados nesta máquina foram: silêncio mínimo 500ms, blocos de até 30s, padding
   de 200ms, contexto zerado (`-mc 0 -sns -vsd 500 -vmsd 30 -vp 200`) — já embutidos
   em `gpu_transcribe.py` quando `--vad-model` é passado. Sempre teste um trecho curto
   isolado antes de rodar o áudio completo.

## Setup: diarização via pyannote + ONNX/DirectML

1. Dependências (além do `pyannote.audio` normal):
   ```bash
   pip install onnxruntime-directml onnxscript
   ```
   (`onnxruntime-directml` substitui o `onnxruntime` padrão pelo pacote com suporte a
   qualquer GPU DirectX12; não precisa dos dois instalados.)
2. Exportar o modelo de embeddings (só precisa rodar de novo se o pyannote atualizar
   o checkpoint `pyannote/speaker-diarization-3.1`):
   ```bash
   python gpu_diar/export_embedding.py
   ```
   Isso gera `gpu_diar/wespeaker_resnet34.onnx`. **Detalhe técnico:** não é possível
   exportar o modelo completo de uma vez, porque a extração de features (fbank) usa
   `torch.vmap` sobre uma rotina Kaldi que não traça para ONNX. A solução foi exportar
   só a parte pesada (o ResNet34 + pooling, que já recebe o fbank como entrada) — a
   extração de fbank continua em CPU/PyTorch, é rápida e não é o gargalo.
3. (Opcional, recomendado) validar a saída antes de confiar nela:
   ```bash
   python gpu_diar/verify_onnx.py
   ```
   Deve mostrar `Max abs diff: 0.000000` entre PyTorch(CPU) e ONNX(DirectML).
4. Usar em qualquer pipeline pyannote:
   ```python
   import os
   from pyannote.audio import Pipeline
   from gpu_diar.onnx_embedding_wrapper import swap_embedding_to_onnx_gpu

   pipeline = Pipeline.from_pretrained(
       "pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"]
   )
   swap_embedding_to_onnx_gpu(pipeline, "gpu_diar/wespeaker_resnet34.onnx")
   diarization = pipeline(audio_path)  # agora ~11x mais rapido na etapa de embeddings
   ```

   **Atenção à versão do pyannote.audio:** na versão 4.0.7, `pipeline(audio)` retorna um
   objeto `DiarizeOutput`, não mais um `Annotation` direto. Os segmentos ficam em
   `resultado.speaker_diarization.itertracks(yield_label=True)`. Isso mudou entre
   versões — confirme sempre no changelog do pyannote antes de rodar um áudio longo,
   porque esse tipo de mudança de API só aparece no fim do processamento (depois da
   parte cara/lenta), quando é tarde demais para evitar o retrabalho.

## `transcribe.py` (pipeline completo)

Roda diarização (GPU se `gpu_diar/wespeaker_resnet34.onnx` existir, senão CPU) e depois
transcrição: se já existir `transcript_words.json` no diretório de saída (por exemplo,
gerado pelo `whisper_gpu/gpu_transcribe.py`), pula a transcrição e reusa esse arquivo;
senão, transcreve via `faster-whisper` em CPU. Sempre reporta progresso incremental em
`status.json` e `progress.txt` no diretório de saída (útil para acompanhar em tempo real
via um dashboard HTML simples, por exemplo).

```bash
python transcribe.py --audio caminho/do/audio.mp3 --out-dir pasta_de_saida
```

Opcionais: `--onnx` aponta para outro modelo de embeddings (o padrão é
`gpu_diar/wespeaker_resnet34.onnx`; se o arquivo não existir, a diarização roda em CPU) e
`--duration` informa a duração em segundos, usada apenas se a detecção automática falhar.

**Cuidados de concorrência**, se rodar diarização (CPU/GPU) e transcrição GPU em paralelo
apontando para a mesma pasta de saída:
- Cada processo deve escrever em seu próprio arquivo `.tmp` antes do `os.replace` atômico
  (com nome único por PID) — no Windows, dois processos tentando substituir o mesmo
  arquivo `status.json.tmp` ao mesmo tempo geram `PermissionError: [WinError 5]` e podem
  derrubar o processo inteiro. Sempre envolva a escrita de status em try/except com
  retries; nunca deixe uma falha de *logging* derrubar o trabalho pesado.
- Evite scripts "watcher" que matam processos automaticamente ao detectar um arquivo
  de progresso — se dois pipelines rodam em paralelo apontando pro mesmo diretório,
  prefira deixar cada um seguir seu curso e só combinar os resultados no final.

## Combinando diarização + transcrição

Depois de ter `diarization.json` (segmentos com falante) e `transcript_words.json`
(segmentos de texto com timestamps), o transcrito final é gerado atribuindo a cada
segmento de texto o falante com maior sobreposição temporal no `diarization.json`:

```bash
python merge_transcript.py --dir pasta_de_saida --title "Nome do audio"
```

Gera `transcricao_final.md` na mesma pasta, com falas consecutivas do mesmo falante
agrupadas em parágrafos.
