# Exemplo de orquestração: espera a diarização (CPU/GPU) terminar e então dispara a
# transcrição GPU (whisper.cpp/Vulkan) apontando pro mesmo diretório de saída.
#
# ATENÇÃO: a versão original usada nesta máquina também matava automaticamente
# qualquer processo `transcribe.py` detectado assim que `diarization.json` aparecia.
# Isso é arriscado se os dois pipelines rodarem em paralelo — o processo pode ser
# encerrado antes de terminar de escrever seus próprios arquivos. Prefira deixar cada
# processo terminar seu próprio trabalho e só then combinar os resultados (ver
# merge_transcript.py de exemplo no README). O trecho de Stop-Process foi mantido
# comentado abaixo só como referência do que foi realmente executado.

$ErrorActionPreference = 'Stop'
$jobDir = '<CAMINHO_DA_PASTA_DE_SAIDA>'
$diarization = Join-Path $jobDir 'diarization.json'
$transcript = Join-Path $jobDir 'transcript_words.json'
$handoffLog = Join-Path $jobDir 'gpu_handoff.log'
$gpuScript = '<CAMINHO_PARA_gpu_transcribe.py>'

"[$(Get-Date -Format HH:mm:ss)] Aguardando a diarizacao terminar..." | Add-Content -LiteralPath $handoffLog

while (-not (Test-Path -LiteralPath $diarization)) {
    Start-Sleep -Milliseconds 200
}

"[$(Get-Date -Format HH:mm:ss)] Diarizacao encontrada; transferindo para a GPU." | Add-Content -LiteralPath $handoffLog

# --- Comportamento original (arriscado, ver nota acima) ---
# $cpuProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
#     Where-Object {
#         $_.Name -eq 'python.exe' -and
#         $_.CommandLine -match '(^|[\\/\s"])transcribe\.py(["\s]|$)' -and
#         $_.CommandLine -notmatch 'gpu_transcribe\.py'
#     }
# foreach ($cpuProcess in $cpuProcesses) {
#     Stop-Process -Id $cpuProcess.ProcessId -Force -ErrorAction SilentlyContinue
# }

if (Test-Path -LiteralPath $transcript) {
    "[$(Get-Date -Format HH:mm:ss)] Uma transcricao ja existe; GPU nao iniciada." | Add-Content -LiteralPath $handoffLog
    exit 0
}

$arguments = @(
    $gpuScript,
    '--audio', '<CAMINHO_DO_AUDIO.mp3>',
    '--out-dir', $jobDir,
    '--work-dir', '<CAMINHO_DE_TRABALHO_TEMPORARIO>',
    '--ffmpeg', '<CAMINHO_PARA_ffmpeg.exe>',
    '--whisper', '<CAMINHO_PARA_whisper-cli.exe>',
    '--model', '<CAMINHO_PARA_ggml-large-v3.bin>',
    '--vad-model', '<CAMINHO_PARA_ggml-silero-v6.2.0.bin>',
    '--duration', '<DURACAO_DO_AUDIO_EM_SEGUNDOS>'
)

& 'python.exe' @arguments *>> $handoffLog
$exitCode = $LASTEXITCODE
"[$(Get-Date -Format HH:mm:ss)] Processo GPU terminou com codigo $exitCode." | Add-Content -LiteralPath $handoffLog
exit $exitCode
