# F3 — Optional RNNoise residual denoising: A/B result

> **This is the F3 evidence record.** The verdict and the headline A/B numbers
> are folded into [06-voice.md](06-voice.md) §F3, which is the doc to read first;
> the full sweep, install path, and fixture design stay here. [README.md](README.md)
> and [TASKBOARD.md](../../TASKBOARD.md) carry the same verdict.

**Date:** 2026-08-26 · **Device:** Jetson Orin Nano Super 8GB, L4T R36.4.4, JetPack 6.x, aarch64, Ubuntu 22.04, headless
**Gate:** `./scripts/bringup/f3_rnnoise_ab.sh` (installs RNNoise if needed, then runs the sweep)
**Report:** `data/bringup/f3_rnnoise_ab.json` · **Audio:** `data/audio/f3/` (128 WAVs) · both gitignored, re-run to regenerate

## Verdict

**DROP RNNoise.** Do not put it in the voice pipeline, and do not pin it in
`jetbot_agent/requirements.txt`.

RNNoise is the better *denoiser* on every signal metric — 270× noise reduction
against the APM's 8.2×, and +4.97 dB segmental SNR against the APM's +0.76 dB —
and it still makes the robot **worse at hearing**. Placed after the WebRTC APM as
a residual denoiser it raises FastConformer word error rate on real noisy speech
from **0.006 to 0.253**, a 42× regression, peaking at **0.889 WER** on the fixture
that most resembles this robot's actual noise floor (fan hum plus motor whine at
5 dB SNR). It also costs 15× the APM's CPU and adds ~52 ms of buffering delay.

This is the textbook denoiser failure mode, measured rather than assumed: the
perceptual metrics improve while the acoustic model's input distribution is
destroyed. The APM alone is both cheaper and more accurate, so there is nothing
to trade off.

WebRTC APM stays the required front end. **RNNoise was never a candidate to
replace AEC and the measurements confirm it cannot be one:** on the F2 echo
fixture the APM cancels 2137× and RNNoise 4.5×.

## 1. Install path taken

`pyrnnoise==0.4.3`, installed **with `--no-deps`** from its
`manylinux2014_aarch64` wheel, plus `soxr==1.1.0` for resampling. No source
build, no autotools, no `sudo`.

```bash
./scripts/bringup/install_rnnoise.sh    # idempotent; also called by the gate wrapper
```

Why this path:

| Option | Outcome |
| --- | --- |
| `pyrnnoise` wheel (aarch64) | **Taken.** 13.3 MB pure-Python wheel bundling a prebuilt `librnnoise.so` with the model weights compiled in. Reachable from `pypi.org`. |
| Build `xiph/rnnoise` from source | Not needed. Would also have been at risk: upstream's `download_model.sh` fetches weights from `media.xiph.org`, which is **not** on the sandbox allowlist, and `libtool` is absent from this board. |
| Any Hugging Face-hosted binding | Unreachable — `huggingface.co` is not allowlisted, same constraint F4 hit. |

Two deliberate deviations from a plain `pip install pyrnnoise`:

- **`--no-deps`.** `pyrnnoise` declares `audiolab` (PyAV), `matplotlib`, `click`
  and `tqdm`. Those exist only for its file/CLI helpers. Pulling PyAV and
  matplotlib onto an 8 GB robot to reach a 480-sample C function is not
  justifiable, so F3 drives the bundled library directly.
- **The binding is loaded by file path.** `pyrnnoise/__init__.py` imports the
  heavy path, so `import pyrnnoise` fails under `--no-deps`. The low-level
  `pyrnnoise/rnnoise.py` ctypes shim needs only `numpy` + `ctypes`, so
  `load_rnnoise_shim()` in the gate loads that single file directly. Upstream's
  own binding is reused rather than rewritten, so the C ABI signatures are theirs.

Net addition to `.venv`: two packages (`pyrnnoise`, `soxr`), ~15 MB, no
transitive dependencies beyond `numpy`, which was already present.

## 2. Constraints and what the resampling actually costs

RNNoise is hard-wired to **48 kHz mono, 480-sample (10 ms) frames**; the confirmed
`rnnoise_get_frame_size()` on this build is 480. The pipeline is 16 kHz, so every
RNNoise configuration below is really:

```
16 kHz → soxr (HQ) → 48 kHz → RNNoise 480-sample frames → soxr (HQ) → 16 kHz
```

One good piece of news, and one bad one. Both measured, not reasoned:

**The rate conversion is nearly free, and adds no delay.** 16 kHz → 48 kHz is an
exact 1:3 ratio, and 10 ms at 16 kHz (160 samples) maps to exactly one 480-sample
RNNoise frame — a 1.0 frame-per-frame match, so the existing 10 ms APM framing
carries straight through with no extra fragmentation. Measured group delay is
**0.00 ms for the soxr round trip** and **0.00 ms for RNNoise itself** (RNNoise's
overlap-add is causal). The resampler's CPU cost is **RTF 0.00087**, i.e. 0.5% of
the RNNoise stage it feeds.

**The framing glue is not free.** `soxr.ResampleStream` does not emit one
480-sample block per input frame; it buffers and emits in bursts (nothing for
roughly two thirds of frames, then 1468–2140 samples at once). Bridging that to
RNNoise's fixed frame size needs a reframing buffer, and counting the emission
schedule sample-exactly gives **51.8 ms of added delay typically, 67.1 ms worst
case** — five to seven times the 10 ms frame budget the F6 duplex pipeline is
built around.

That last number is an artifact of the off-the-shelf pieces, not of RNNoise, and
it is fixable: since both RNNoise and the soxr filters measure 0 ms, a fixed
integer 1:3 polyphase resampler emitting exactly 480 samples per 10 ms frame
would drive the chain toward ~0 ms added delay. But that is custom DSP we would
own and maintain, and per §4 there is no accuracy benefit to buy with it. It is
recorded here so the number is not mistaken for an intrinsic RNNoise property.

Throughput is fine either way — 0 of 662 frames exceeded the 10 ms budget — but
the per-frame cost is bursty for the same reason: mean 1.90 ms, p95 5.67 ms, max
9.39 ms of a 10 ms budget. A front end that occasionally consumes 94% of its
real-time budget is uncomfortable next to a camera, a motor watchdog, and an
agent loop.

## 3. The A/B

Same fixtures, same APM code, and the same RMS/peak helpers as F2, so the
comparison is exact rather than approximate. Two independent checks confirm the
parity: `apm_ns` on the F2 noise fixture reproduces
**8.224574897331724** and full-APM clean peak reproduces **0.97442626953125** —
both bit-identical to the values published in F2.

Configurations (`+rnnoise` always means RNNoise **after** the APM):

| Label | Chain |
| --- | --- |
| `raw` | unprocessed |
| `apm_ns` | APM, NS level 3 + HPF only, AGC off — the exact F2 noise-reduction configuration |
| `apm` | APM, NS + HPF + AEC, **AGC off** — headline row; AGC off removes a gain confound from every metric |
| `apm+rnnoise` | `apm` → RNNoise |
| `apm_agc` | full production APM: NS + HPF + AEC + **AGC** |
| `apm_agc+rnnoise` | `apm_agc` → RNNoise |
| `rnnoise` | RNNoise alone, for reference only |

### Signal and cost

Noise/echo columns are the F2 fixtures. Speech columns are averaged over the ten
real-speech noisy fixtures (§4). `Δ segSNR` is improvement over `raw`.

| Config | Noise-only ↓ | Echo-only ↓ | segSNR clean (dB) | Δ segSNR noisy (dB) | RTF | CPU RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `raw` | 1.0× | 1.0× | — | 0.00 | 0.0000 | 0.0000 |
| `apm_ns` | 8.2× | 10.4× | 3.98 | +0.80 | 0.0026 | 0.0026 |
| `apm` | 8.9× | **2137×** | 4.01 | +0.76 | 0.0126 | 0.0126 |
| `apm+rnnoise` | **73.2×** | 20524× | 2.74 | +0.51 | 0.1982 | 0.1975 |
| `apm_agc` | 1.6× | 297× | 4.00 | +1.10 | 0.0130 | 0.0130 |
| `apm_agc+rnnoise` | 29.2× | 6829× | 2.69 | +1.37 | 0.1989 | 0.1982 |
| `rnnoise` | **270×** | **4.5×** | 11.87 | **+4.97** | 0.1859 | 0.1854 |

Reading it:

- **RNNoise wins the denoising contest outright.** 270× on noise-only versus the
  APM's 8.2×; +4.97 dB segSNR versus +0.76 dB. If the decision rested on these
  columns it would be an easy keep. §4 is why it does not.
- **RNNoise is not an echo canceller.** 4.5× on the echo fixture against the APM's
  2137× — 475× worse. The `apm+rnnoise` echo figure (20524×) is the APM's AEC
  doing the work and RNNoise attenuating what little residual is left; it is not
  evidence of echo cancellation. `apm_agc`'s lower 297× is the AGC pulling the
  residual floor back up, exactly as F2 saw.
- **RNNoise costs 15× the APM.** RTF 0.198 versus 0.013. Essentially all of it is
  inside `librnnoise.so` — 1.82 ms per 480-sample frame, RTF 0.182 measured in a
  tight C-call loop with pre-allocated frames, so it is not Python or ctypes
  overhead. That is slow for RNNoise, and plausibly a property of this generic
  `manylinux2014` build rather than of the algorithm; a source build with
  `-O3 -mcpu=native` might do considerably better. Untested, and it does not
  change the verdict, which rests on accuracy rather than cost.
- **The APM's lower segSNR does not mean worse ASR** — see below. segSNR is a
  waveform-fidelity measure and the two ranks invert against WER, which is
  precisely the trap F3 exists to catch.
- Peak RSS for the whole sweep process was 97 MiB; `tegrastats` showed
  2242/7620 MB RAM, `GR3D_FREQ 0%`, 0 MB swap touched. Memory was never a factor.

## 4. Downstream ASR — the decision-relevant number

Recognizer: F4's recommended production configuration — sherpa-onnx
`OfflineRecognizer.from_nemo_ctc`, NeMo FastConformer CTC int8 ONNX, CPU, 2
threads. WER is computed on case-folded, punctuation-stripped words, same helper
as F4. Decode RTF stayed at 0.0435–0.0437 and peak RSS at 432–503 MiB across all
seven configurations, in line with F4's 324–466 MiB — the recognizer itself is
not what changes here, only what it is fed.

**Audio provenance, stated plainly.** Speech is two real LibriSpeech utterances
(6.6 s and 16.7 s) shipped inside the F4 model archives, with their reference
transcripts, so WER is measured against real ground truth. The **noise is
synthetic**, additively mixed at a measured SNR: white Gaussian, and a
"robot" noise built from 120 Hz fan-hum harmonics plus a frequency-modulated
2.8/5.6 kHz motor whine and broadband. **No real noisy recording with
ground-truth text exists yet** — nothing under `data/audio/` is a microphone
capture; the F2 fixtures are synthetic and the F4 fixtures are clean archive
audio. Five conditions × two utterances = ten noisy fixtures, plus the two clean
originals.

Mean WER, and the worst single fixture:

| Config | WER clean | WER noisy (mean of 10) | Worst fixture |
| --- | --- | --- | --- |
| `raw` | 0.010 | 0.040 | 0.111 |
| `apm_ns` | 0.021 | **0.006** | 0.056 |
| `apm` | 0.031 | **0.006** | 0.056 |
| `apm+rnnoise` | 0.042 | **0.253** | **0.889** |
| `apm_agc` | 0.010 | **0.008** | 0.056 |
| `apm_agc+rnnoise` | 0.042 | 0.119 | 0.444 |
| `rnnoise` | 0.021 | 0.085 | 0.278 |

Per fixture, `apm` versus `apm+rnnoise`:

| Fixture | raw | `apm` | `apm+rnnoise` | `apm_agc` | `apm_agc+rnnoise` | `rnnoise` |
| --- | --- | --- | --- | --- | --- | --- |
| libri0 white 5 dB | 0.056 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| libri0 white 0 dB | 0.111 | 0.056 | 0.111 | 0.056 | 0.111 | 0.222 |
| libri0 robot 10 dB | 0.000 | 0.000 | 0.056 | 0.000 | 0.000 | 0.000 |
| libri0 robot 5 dB | 0.000 | 0.000 | **0.889** | 0.000 | 0.222 | 0.056 |
| libri0 robot 0 dB | 0.111 | 0.000 | **0.722** | 0.000 | 0.444 | 0.278 |
| libri1 white 5 dB | 0.062 | 0.000 | 0.083 | 0.000 | 0.062 | 0.042 |
| libri1 white 0 dB | 0.042 | 0.000 | 0.062 | 0.021 | 0.062 | 0.062 |
| libri1 robot 10 dB | 0.000 | 0.000 | 0.083 | 0.000 | 0.042 | 0.062 |
| libri1 robot 5 dB | 0.000 | 0.000 | 0.083 | 0.000 | 0.062 | 0.062 |
| libri1 robot 0 dB | 0.021 | 0.000 | **0.438** | 0.000 | 0.188 | 0.062 |

What this says:

1. **The APM earns its place as the required front end.** It cuts noisy-speech
   WER from 0.040 to 0.006 — a 6.7× improvement over unprocessed audio, and it
   never loses to `raw` on any noisy fixture. This is independent evidence for
   the architecture decision, not just a restatement of it.
2. **RNNoise after the APM is a regression, not a refinement.** 0.253 versus
   0.006 mean, and it is worse than `raw` on eight of ten fixtures. The single
   worst case, `libri0_robot_5db`, is a configuration where the APM alone scores
   a perfect 0.000:

```
ref:              AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
apm         0.000 After early nightfall, the yellow lamps would light up here and there the squalid quarter of the brothels
apm+rnnoise 0.889 After early nightfll what was coor
rnnoise     0.056 After early nightfall the yellow lamps would light up here and there the squaallid quarter of the brothels
```

3. **The damage is caused by cascading, not by RNNoise itself.** RNNoise *alone*
   scores 0.085, and 0.056 on the fixture where the cascade scores 0.889. Two
   aggressive suppressors in series over-attenuate: `apm+rnnoise` output RMS on
   that fixture is 0.0122 against `apm`'s 0.0327, a further 2.7× of attenuation
   that takes speech with it. Its segSNR also *drops* from 0.37 dB to −0.82 dB, so
   even the perceptual metric agrees on this fixture — the cascade is worse by
   both measures, and only the aggregate perceptual average flattered it.
4. **The failure is worst on exactly the noise this robot has.** The fan/motor
   fixtures fail catastrophically (0.889, 0.722, 0.438) while white noise stays
   near 0.1. Tonal, structured noise is what a JetBot's own chassis produces, and
   that is where RNNoise most confidently subtracts the wrong thing.
5. **AGC between the two stages masks the problem without fixing it.**
   `apm_agc+rnnoise` (0.119) beats `apm+rnnoise` (0.253) because the AGC restores
   level before RNNoise's second pass. It is still ~15× worse than `apm_agc`
   alone (0.008). Do not read this as a viable configuration.
6. **RNNoise also slightly hurts clean speech**: 0.042 against 0.010 for
   `apm_agc`. Small and from only two clean fixtures, so weak evidence — but it
   points the same way as everything else.

## 5. Gate outcome

**F3 PASSES as a gate** on the terms 06-voice.md sets: "reproducible A/B
artifacts and measurements exist; adopting or rejecting RNNoise is documented."
The decision is **reject**. F3 was never required for F4 or F5, and neither is
blocked.

Actions taken: none beyond this document, the gate script, and the `.gitignore`
entry. `jetbot_agent/requirements.txt` already leaves RNNoise unpinned and
described as "an optional F3 benchmark dependency" — that stays true, and F3's
answer is that it should never become a pinned one. `pyrnnoise` and `soxr` remain
in `.venv` so the gate stays re-runnable; because they are not in
`requirements.txt` they will not reach a production install.

`soxr` is worth keeping in mind independently of RNNoise — it is a 210 KB wheel,
depends only on numpy, and measured RTF 0.00087 for a 16k↔48k round trip. If any
later stage needs resampling, it is a good default. That is a separate decision
from this one.

## 6. Caveats and open items

- **No real captured noisy speech.** The strongest possible version of this test
  is the robot's own microphone recording a scripted utterance with its own fan
  and motors running. That fixture does not exist. The verdict rests on synthetic
  noise mixed onto real speech at measured SNRs. The direction of the result is
  large (42×) and consistent across two utterances, two noise families, and three
  SNRs, so it is very unlikely to reverse — but the *magnitude* on real captured
  audio is unmeasured. Re-run this gate after F1 produces a real noisy capture.
- **The APM implementation used here is F4's preserved HEAD copy**,
  `data/models/f4/_apm_head/audio_preprocessor.py`. The working-tree
  `jetbot_agent/audio/audio_preprocessor.py` is currently reverted to a
  `StageNotReady` stub by an unrelated in-flight edit — the same condition F4
  recorded. The gate tries the package module first and falls back, recording
  which it used in `apm_provenance`. The exact F2 parity numbers above confirm
  the fallback is the real F2 code.
- **The 0.182 RNNoise RTF may be pessimistic**, being a generic
  `manylinux2014_aarch64` build. Not re-tested with an optimised source build,
  because CPU cost is not what decides this.
- **The 52 ms buffering delay is an implementation artifact** of
  `soxr.ResampleStream`'s bursty output, not an RNNoise property. RNNoise and the
  soxr filters each measured 0 ms.
- **AGC off for the headline rows.** `apm` and `apm+rnnoise` run with AGC
  disabled so that RMS and SNR comparisons are not confounded by gain, matching
  F2's approach for its NS/AEC ratios. `apm_agc` rows carry the full production
  front end.
- **No live duplex, no ALSA, no speaker.** Offline gate only. F1's ALSA identity
  and mixer baseline still applies; sequential capture-then-playback remains the
  rule until F6.

## Files

| Path | What |
| --- | --- |
| `scripts/bringup/f3_rnnoise_ab.py` | the A/B gate: fixtures, signal metrics, latency decomposition, ASR sweep |
| `scripts/bringup/f3_rnnoise_ab.sh` | wrapper, matching the other bring-up gates |
| `scripts/bringup/install_rnnoise.sh` | idempotent RNNoise + soxr install with a smoke test |
| `data/bringup/f3_rnnoise_ab.json` | full report (gitignored) |
| `data/bringup/f3_run.log` | captured gate stdout, including the formatted tables (gitignored) |
| `data/audio/f3/` | 128 input and processed WAVs, one per fixture × config (gitignored) |
| `data/models/f3/` | cached `pyrnnoise` wheel (gitignored via a new `.gitignore` entry) |
