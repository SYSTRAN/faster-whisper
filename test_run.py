"""End-to-end benchmark for WhisperModel.transcribe_batch.

Usage:
  # Basic run with a fixed batch size:
    python test_run.py --batch-size 8

  # Sweep to find the sweet spot (tries batch sizes 1,2,4,...,max-batch-size):
    python test_run.py --batch-size 64 --search-from-batch-size 1

  # Custom sweep start:
    python test_run.py --batch-size 128 --search-from-batch-size 8

  # Compare beam_size=1 vs 5:
    python test_run.py --batch-size 32 --search-from-batch-size 1 --beam-size 1
"""

import argparse
import time

from faster_whisper import WhisperModel, decode_audio

parser = argparse.ArgumentParser()
parser.add_argument("--audio-path", default="data/short_heb.m4a")
parser.add_argument("--language", default="he")
parser.add_argument("--model", default="yoad/whisper-tiny-v2-ct2")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--beam-size", type=int, default=5,
                    help="Beam size for decoding (default: 5). Try 1 to isolate beam overhead.")
parser.add_argument(
    "--search-from-batch-size",
    type=int,
    default=None,
    help="When set, sweep batch sizes from this value up to --batch-size "
    "(doubling each step) and report the sweet spot.",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=1,
    help="Number of warmup runs before timing (default: 1).",
)
parser.add_argument(
    "--repeats",
    type=int,
    default=3,
    help="Number of timed runs to average (default: 3).",
)
args = parser.parse_args()

# Instrumentation phase labels (in pipeline order)
PHASE_LABELS = [
    "1_decode_audio",
    "1_feature_extract",
    "2_language_detect",
    "3_tokenizer_prompt",
    "4_pad_stack",
    "5_encode",
    "6_multilingual_detect",
    "7_generate",
    "8_postprocess",
]

# Extra generate sub-phases for the decoder deep-dive
GENERATE_SUB_LABELS = [
    "7a_generate_1step",
    "7b_generate_full",
]


def time_sequential(model, audio, n, language, beam_size, repeats):
    """Run model.transcribe n times sequentially, return best wall time."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            segs, _ = model.transcribe(audio, language=language, beam_size=beam_size)
            list(segs)  # drain the generator
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def time_batch_instrumented(model, audio, n, language, beam_size, repeats):
    """Run model.transcribe_batch with instrumentation, return (best_time, best_timings)."""
    batch = [audio] * n
    best_time = float("inf")
    best_timings = {}
    for _ in range(repeats):
        t0 = time.perf_counter()
        _result, timings = model.transcribe_batch(
            batch, language=language, beam_size=beam_size, _instrumentation=True,
        )
        elapsed = time.perf_counter() - t0
        if elapsed < best_time:
            best_time = elapsed
            best_timings = timings
    return best_time, best_timings


def sweep_batch_sizes(start, end):
    """Generate batch sizes: start, start*2, start*4, ..., end."""
    sizes = []
    s = start
    while s <= end:
        sizes.append(s)
        s *= 2
    if sizes[-1] != end:
        sizes.append(end)
    return sizes


def print_phase_breakdown(timings, batch_size):
    """Print a per-phase timing breakdown for one batch size."""
    total = timings.get("total", sum(timings.get(p, 0) for p in PHASE_LABELS))
    print(f"  {'phase':<25} {'time (ms)':>10} {'%':>7}  {'per-item (ms)':>13}")
    print(f"  {'-'*58}")
    for phase in PHASE_LABELS:
        t = timings.get(phase, 0)
        pct = (t / total * 100) if total > 0 else 0
        per_item = (t / batch_size) * 1000
        print(f"  {phase:<25} {t*1000:>10.1f} {pct:>6.1f}%  {per_item:>13.1f}")
    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<25} {total*1000:>10.1f} {100.0:>6.1f}%  {total/batch_size*1000:>13.1f}")


def print_sweep_phase_table(all_timings, sizes):
    """Print a phase x batch-size matrix showing where time goes as batch grows."""
    print("\n=== Phase breakdown across batch sizes (ms) ===\n")

    # Header
    header = f"  {'phase':<25}"
    for n in sizes:
        header += f" {'B='+str(n):>9}"
    print(header)
    print(f"  {'-' * (25 + 10 * len(sizes))}")

    for phase in PHASE_LABELS:
        row = f"  {phase:<25}"
        for n in sizes:
            t = all_timings[n].get(phase, 0) * 1000
            row += f" {t:>9.1f}"
        print(row)

    # Total row
    row = f"  {'TOTAL':<25}"
    for n in sizes:
        t = all_timings[n].get("total", 0) * 1000
        row += f" {t:>9.1f}"
    print(f"  {'-' * (25 + 10 * len(sizes))}")
    print(row)

    # Per-item row
    row = f"  {'TOTAL / item':<25}"
    for n in sizes:
        t = all_timings[n].get("total", 0) * 1000 / n
        row += f" {t:>9.1f}"
    print(row)

    # Percentage table
    print(f"\n=== Phase share of total (%) ===\n")
    header = f"  {'phase':<25}"
    for n in sizes:
        header += f" {'B='+str(n):>9}"
    print(header)
    print(f"  {'-' * (25 + 10 * len(sizes))}")

    for phase in PHASE_LABELS:
        row = f"  {phase:<25}"
        for n in sizes:
            total = all_timings[n].get("total", 1)
            t = all_timings[n].get(phase, 0)
            pct = (t / total * 100) if total > 0 else 0
            row += f" {pct:>8.1f}%"
        print(row)

    # Scaling analysis: how each phase scales relative to batch=min
    print(f"\n=== Phase scaling factor (relative to B={sizes[0]}) ===\n")
    header = f"  {'phase':<25}"
    for n in sizes:
        header += f" {'B='+str(n):>9}"
    print(header)
    print(f"  {'-' * (25 + 10 * len(sizes))}")

    base_size = sizes[0]
    for phase in PHASE_LABELS:
        row = f"  {phase:<25}"
        base_t = all_timings[base_size].get(phase, 0)
        for n in sizes:
            t = all_timings[n].get(phase, 0)
            if base_t > 0.0001:  # avoid division by near-zero
                factor = t / base_t
                row += f" {factor:>8.1f}x"
            else:
                row += f" {'~0':>9}"
        print(row)

    # Show ideal linear scaling for reference
    row = f"  {'(ideal linear)':<25}"
    for n in sizes:
        row += f" {n/base_size:>8.1f}x"
    print(row)


def print_generate_deep_dive(all_timings, sizes):
    """Print the decoder deep-dive: 1-step probe, full generate, token stats."""
    print("\n=== Decoder (generate) deep-dive ===\n")

    # 1-step vs full generate timing
    header = f"  {'metric':<30}"
    for n in sizes:
        header += f" {'B='+str(n):>9}"
    print(header)
    print(f"  {'-' * (30 + 10 * len(sizes))}")

    # 1-step probe (ms)
    row = f"  {'1-step probe (ms)':<30}"
    for n in sizes:
        t = all_timings[n].get("7a_generate_1step", 0) * 1000
        row += f" {t:>9.1f}"
    print(row)

    # Full generate (ms)
    row = f"  {'full generate (ms)':<30}"
    for n in sizes:
        t = all_timings[n].get("7b_generate_full", 0) * 1000
        row += f" {t:>9.1f}"
    print(row)

    # Autoregressive portion = full - 1step
    row = f"  {'autoregressive portion (ms)':<30}"
    for n in sizes:
        full = all_timings[n].get("7b_generate_full", 0) * 1000
        step1 = all_timings[n].get("7a_generate_1step", 0) * 1000
        row += f" {full - step1:>9.1f}"
    print(row)

    # 1-step as % of full
    row = f"  {'1-step as % of full':<30}"
    for n in sizes:
        full = all_timings[n].get("7b_generate_full", 0)
        step1 = all_timings[n].get("7a_generate_1step", 0)
        pct = (step1 / full * 100) if full > 0 else 0
        row += f" {pct:>8.1f}%"
    print(row)

    print()

    # 1-step scaling
    base_size = sizes[0]
    row = f"  {'1-step scaling':<30}"
    base_t = all_timings[base_size].get("7a_generate_1step", 0)
    for n in sizes:
        t = all_timings[n].get("7a_generate_1step", 0)
        if base_t > 0.0001:
            row += f" {t/base_t:>8.1f}x"
        else:
            row += f" {'~0':>9}"
    print(row)

    # Full generate scaling
    row = f"  {'full generate scaling':<30}"
    base_t = all_timings[base_size].get("7b_generate_full", 0)
    for n in sizes:
        t = all_timings[n].get("7b_generate_full", 0)
        if base_t > 0.0001:
            row += f" {t/base_t:>8.1f}x"
        else:
            row += f" {'~0':>9}"
    print(row)

    # Autoregressive scaling
    row = f"  {'autoregressive scaling':<30}"
    base_full = all_timings[base_size].get("7b_generate_full", 0)
    base_step1 = all_timings[base_size].get("7a_generate_1step", 0)
    base_auto = base_full - base_step1
    for n in sizes:
        full = all_timings[n].get("7b_generate_full", 0)
        step1 = all_timings[n].get("7a_generate_1step", 0)
        auto = full - step1
        if base_auto > 0.0001:
            row += f" {auto/base_auto:>8.1f}x"
        else:
            row += f" {'~0':>9}"
    print(row)

    row = f"  {'(ideal linear)':<30}"
    for n in sizes:
        row += f" {n/base_size:>8.1f}x"
    print(row)

    # Token counts
    print()
    row = f"  {'tokens/item (avg)':<30}"
    for n in sizes:
        avg = all_timings[n].get("7_avg_tokens", 0)
        row += f" {avg:>9.1f}"
    print(row)

    row = f"  {'tokens/item (max)':<30}"
    for n in sizes:
        mx = all_timings[n].get("7_max_tokens", 0)
        row += f" {mx:>9}"
    print(row)

    row = f"  {'effective beam*batch':<30}"
    for n in sizes:
        ebb = all_timings[n].get("7_effective_beam_batch", 0)
        row += f" {ebb:>9}"
    print(row)

    # ms per decode step (approximate): autoregressive_ms / avg_tokens
    row = f"  {'ms/decode step (approx)':<30}"
    for n in sizes:
        full = all_timings[n].get("7b_generate_full", 0) * 1000
        step1 = all_timings[n].get("7a_generate_1step", 0) * 1000
        auto = full - step1
        avg_tok = all_timings[n].get("7_avg_tokens", 1)
        # avg_tok includes prompt tokens -- but all items produce ~same count
        # so this is a rough per-step cost
        ms_per_step = auto / max(avg_tok, 1)
        row += f" {ms_per_step:>9.2f}"
    print(row)


def main():
    model = WhisperModel(args.model)
    audio = decode_audio(args.audio_path)
    duration = audio.shape[0] / model.feature_extractor.sampling_rate
    print(f"Audio: {args.audio_path} ({duration:.2f}s)")
    print(f"Model: {args.model}")
    print(f"Beam size: {args.beam_size}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats} (best-of)")

    # -- warmup --
    for _ in range(args.warmup):
        list(model.transcribe(audio, language=args.language, beam_size=args.beam_size)[0])
        model.transcribe_batch([audio], language=args.language, beam_size=args.beam_size)

    if args.search_from_batch_size is not None:
        # -- sweep mode --
        sizes = sweep_batch_sizes(args.search_from_batch_size, args.batch_size)
        print(f"\nSweeping batch sizes: {sizes}")
        print(f"\n{'batch':>6}  {'sequential':>11}  {'batched':>11}  {'speedup':>8}  {'per-item':>10}")
        print(f"{'size':>6}  {'(s)':>11}  {'(s)':>11}  {'':>8}  {'batch (ms)':>10}")
        print("-" * 60)

        best_speedup = 0.0
        best_size = sizes[0]
        sweep_results = []
        all_timings = {}

        for n in sizes:
            t_seq = time_sequential(model, audio, n, args.language, args.beam_size, args.repeats)
            t_bat, timings = time_batch_instrumented(model, audio, n, args.language, args.beam_size, args.repeats)
            speedup = t_seq / t_bat if t_bat > 0 else float("inf")
            per_item_ms = (t_bat / n) * 1000

            print(f"{n:>6}  {t_seq:>11.3f}  {t_bat:>11.3f}  {speedup:>7.2f}x  {per_item_ms:>10.1f}")
            sweep_results.append((n, t_seq, t_bat, speedup, per_item_ms))
            all_timings[n] = timings

            if speedup > best_speedup:
                best_speedup = speedup
                best_size = n

        print("-" * 60)
        print(f"Sweet spot: batch_size={best_size} ({best_speedup:.2f}x speedup)")

        # find where batching becomes slower than sequential
        crossover = None
        for n, t_seq, t_bat, speedup, _ in sweep_results:
            if speedup < 1.0:
                crossover = n
                break
        if crossover:
            print(f"Crossover (batch slower than sequential): batch_size={crossover}")
        else:
            print("Batching was faster than sequential at all tested sizes.")

        # Print the detailed phase tables
        print_sweep_phase_table(all_timings, sizes)

        # Decoder deep-dive
        print_generate_deep_dive(all_timings, sizes)

        # Per-size detailed breakdown
        for n in sizes:
            print(f"\n--- Detailed breakdown: batch_size={n} ---")
            print_phase_breakdown(all_timings[n], n)

    else:
        # -- single batch size mode --
        n = args.batch_size

        # sequential baseline (actually run it)
        t_seq = time_sequential(model, audio, n, args.language, args.beam_size, args.repeats)

        # show single-transcribe output once for reference
        segs, info = model.transcribe(audio, language=args.language, beam_size=args.beam_size)
        segs = list(segs)
        print(f"\n--- transcription (language={info.language}, p={info.language_probability:.2f}) ---")
        for seg in segs:
            print(f"  [{seg.start:.2f} -> {seg.end:.2f}] {seg.text}")

        # batch with instrumentation
        t_bat, timings = time_batch_instrumented(model, audio, n, args.language, args.beam_size, args.repeats)
        speedup = t_seq / t_bat if t_bat > 0 else float("inf")

        print(f"\n--- benchmark (n={n}, beam_size={args.beam_size}) ---")
        print(f"  sequential ({n}x transcribe): {t_seq:.3f}s")
        print(f"  batched    (transcribe_batch): {t_bat:.3f}s")
        print(f"  speedup: {speedup:.2f}x")

        print(f"\n--- phase breakdown ---")
        print_phase_breakdown(timings, n)

        # sanity check
        results = model.transcribe_batch([audio] * n, language=args.language, beam_size=args.beam_size)
        texts = [" ".join(s.text for s in segs) for segs, _ in results]
        assert len(set(texts)) == 1, "batch results differ across identical inputs"
        print("\n  all batch outputs identical: OK")


if __name__ == "__main__":
    main()
