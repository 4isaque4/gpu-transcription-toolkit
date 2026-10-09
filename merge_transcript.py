"""Combina diarization.json (falantes) + transcript_words.json (texto) num
transcrito final legivel em Markdown, atribuindo a cada trecho de fala o
falante com maior sobreposicao temporal.

Uso:
    python merge_transcript.py --dir <pasta_com_diarization.json_e_transcript_words.json> --title "Meu audio"
"""
import argparse
import json
import os


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def speaker_for(diarization, seg_start, seg_end):
    best_spk, best_ov = None, 0.0
    for d in diarization:
        if d["end"] < seg_start:
            continue
        if d["start"] > seg_end:
            break
        ov = overlap(seg_start, seg_end, d["start"], d["end"])
        if ov > best_ov:
            best_ov, best_spk = ov, d["speaker"]
    return best_spk or "SPEAKER_?"


def fmt_ts(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Pasta com diarization.json e transcript_words.json")
    parser.add_argument("--title", default="Transcricao")
    parser.add_argument("--max-gap", type=float, default=2.0, help="Segundos de silencio maximo para juntar falas consecutivas do mesmo falante")
    args = parser.parse_args()

    with open(os.path.join(args.dir, "transcript_words.json"), encoding="utf-8") as f:
        transcript = json.load(f)
    with open(os.path.join(args.dir, "diarization.json"), encoding="utf-8") as f:
        diarization = json.load(f)
    diarization.sort(key=lambda s: s["start"])

    labeled = []
    for seg in transcript:
        spk = speaker_for(diarization, seg["start"], seg["end"])
        labeled.append({"start": seg["start"], "end": seg["end"], "speaker": spk, "text": seg["text"].strip()})

    grouped = []
    for seg in labeled:
        if grouped and grouped[-1]["speaker"] == seg["speaker"] and seg["start"] - grouped[-1]["end"] < args.max_gap:
            grouped[-1]["end"] = seg["end"]
            grouped[-1]["text"] += " " + seg["text"]
        else:
            grouped.append(dict(seg))

    speakers_used = sorted(set(g["speaker"] for g in grouped))

    lines = [
        f"# {args.title}\n",
        f"Falantes detectados: {len(speakers_used)} ({', '.join(speakers_used)})\n",
        "---\n",
    ]
    for g in grouped:
        lines.append(f"**[{fmt_ts(g['start'])} - {fmt_ts(g['end'])}] {g['speaker']}:** {g['text'].strip()}\n")

    output_md = os.path.join(args.dir, "transcricao_final.md")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Escrito {output_md}")
    print(f"{len(grouped)} paragrafos, {len(speakers_used)} falantes: {speakers_used}")


if __name__ == "__main__":
    main()
