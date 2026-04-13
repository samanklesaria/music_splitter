# Vocal Harmony Separation — Implementation Plan

**Goal:** Train a model to separate individual vocal parts (lead, harmony voices) from jazz and a cappella ensemble recordings.

---

## Model Choice: SepReformer (SepACap variant)

**Why this model:**

- SepACap (Lanzendörfer & Pinkl, 2024) is the only published architecture targeting multi-voice a cappella separation directly. It achieves SOTA on 5 of 6 stems in the JaCappella benchmark.
- It operates in the waveform domain, avoiding phase reconstruction artifacts.
- It handles a variable number of output stems (up to 6), which suits both quartet and larger ensemble configurations.
- The base architecture (SepReformer, NeurIPS 2024) has an open-source implementation at https://github.com/dmlguq456/SepReformer.

**Architecture overview:**

```
Input waveform (mono, 44.1/48 kHz)
  ↓
Learnable encoder (1-D convolution, short kernel)
  ↓
SepReformer blocks
  - Dual-path processing (intra-chunk + inter-chunk attention)
  - SNAKE activations (sinusoidal, better for periodic/harmonic signals)
  - Rotary position embeddings
  ↓
N mask estimation heads (one per output stem)
  ↓
Learnable decoder (transposed convolution)
  ↓
N separated waveforms
```

**Loss function:** Composite loss combining:
- SI-SDR (scale-invariant signal-to-distortion ratio) in time domain
- Multi-resolution STFT loss (spectral convergence + log magnitude) in frequency domain
- Permutation invariant training (PIT) when voice-role labels are unavailable

**Key paper:** [Source Separation for A Cappella Music](https://arxiv.org/abs/2509.26580)

---

## Datasets

| Dataset | Contents | Size | License | Download |
|---|---|---|---|---|
| **JaCappella** | 35 a cappella songs, 6 stems each (lead, soprano, alto, tenor, bass, percussion), 48 kHz WAV | 4.3 GB | Custom (HF terms) | `scripts/download_jacappella.py` |
| **Dagstuhl ChoirSet** | SATB choral recordings, multiple mic types per singer, F0 annotations | 5.1 GB | CC BY 4.0 | `scripts/download_dagstuhl.py` |
| **MUSDB18-HQ** | 150 pop/rock songs, 4 stems (vocals, drums, bass, other), 44.1 kHz WAV | 22.7 GB | Non-commercial | `scripts/download_musdb18hq.py` |
| **Acappella** | ~46 hrs solo a cappella singing videos from YouTube | Variable | CC BY 4.0 (metadata) | `scripts/download_acappella.py` |

**Primary training data:** JaCappella + Dagstuhl ChoirSet
**Pretraining data:** MUSDB18-HQ (vocal/accompaniment separation, then fine-tune)
**Auxiliary data:** Acappella (solo voice modeling, data augmentation source)

### Data augmentation (critical — primary datasets are small)

Following SepACap's power-set augmentation strategy:

1. **Power-set mixing:** From N isolated stems, generate all possible subsets as training mixtures. This expands 35 JaCappella songs to ~105k training pairs.
2. **Pitch shifting:** ±2 semitones per stem independently before mixing.
3. **Time stretching:** 0.9x–1.1x per stem.
4. **Random gain:** ±6 dB per stem.
5. **Synthetic mixture creation:** Combine isolated stems from different songs to create novel mixtures.
6. **Room impulse response (RIR) convolution:** Simulate different acoustic environments.

---

## Implementation Phases

### Phase 0 — Setup & Data (Week 1)

- [ ] Set up project with `uv`, `equinox`, and `tensorboardX`. 
- [ ] Run `scripts/download_all.sh` to fetch all datasets
- [ ] Write dataset loaders for JaCappella and DCS formats
- [ ] Implement power-set augmentation pipeline
- [ ] Verify audio loading, resampling to uniform sample rate (44.1 kHz)
- [ ] Compute dataset statistics (duration, loudness distribution, pitch ranges)

### Phase 1 — Baseline Model (Weeks 2–3)

- [ ] Port SepReformer architecture from https://github.com/dmlguq456/SepReformer
- [ ] Replace activations with SNAKE (periodic activation, better for harmonics)
- [ ] Implement composite loss (SI-SDR + multi-resolution STFT)
- [ ] Implement PIT (permutation invariant training) loss wrapper
- [ ] Train on JaCappella with 4-stem output (lead + 3 harmony groups)
- [ ] Evaluate: SI-SDR improvement (SI-SDRi), SDR, SAR, SIR on held-out JaCappella test set
- [ ] Target baseline: ≥8 dB SI-SDRi (SepACap reports ~10 dB on JaCappella)

### Phase 2 — Pretraining + Fine-tuning (Weeks 4–5)

- [ ] Pretrain on MUSDB18-HQ for vocal/accompaniment separation (2-stem: vocals vs. rest)
- [ ] Fine-tune the pretrained model on JaCappella + DCS for multi-voice separation
- [ ] Compare against Phase 1 baseline (expect +1–2 dB SI-SDRi from pretraining)
- [ ] Add DCS data: adapt loader for DCS mic configurations (use headset mic as cleanest signal)

### Phase 3 — Jazz-Specific Tuning (Weeks 6–7)

- [ ] Curate jazz-specific subset from JaCappella (jazz/bossa nova songs)
- [ ] Collect additional jazz vocal ensemble recordings if available (Spotify stems, licensed content)
- [ ] Implement voice-role conditioning via FiLM layers:
  - Add voice-type embedding (lead / harmony-high / harmony-mid / harmony-low)
  - Inject via feature-wise linear modulation in decoder
- [ ] Evaluate whether conditioning improves over PIT for jazz material
- [ ] Optional: add F0 conditioning using CREPE or PYIN pitch tracking

### Phase 4 — Evaluation & Optimization (Week 8)

- [ ] Full evaluation suite:
  - SI-SDRi, SDR, SAR, SIR per stem
  - PESQ and STOI for perceptual quality
  - Informal listening tests
- [ ] Model optimization:
  - Knowledge distillation to smaller model for inference
- [ ] Error analysis: identify failure modes (unison passages, voice crossing, vibrato overlap)

---

## Compute Requirements

| Phase | GPU Hours (est.) | Hardware |
|---|---|---|
| Phase 1 (baseline) | ~50 hrs | 1× A100 40GB |
| Phase 2 (pretrain + fine-tune) | ~100 hrs | 1× A100 40GB |
| Phase 3 (jazz tuning) | ~30 hrs | 1× A100 40GB |
| Phase 4 (eval + distillation) | ~20 hrs | 1× A100 40GB |

Total: ~200 GPU hours. Can be done on a single A100 in ~8 days continuous or ~2 weeks with experimentation.

---

## Evaluation Metrics

- **SI-SDRi** (primary): Scale-invariant SDR improvement over mixture. Target ≥10 dB.
- **SDR / SAR / SIR**: Standard BSS Eval metrics via `museval`.
- **PESQ**: Perceptual quality of separated vocals.
- **STOI**: Short-time objective intelligibility.

---

## Relevant Papers

1. **SepACap** — Lanzendörfer, Pinkl, Grotschla (2024). "Source Separation for A Cappella Music." [arxiv:2509.26580](https://arxiv.org/abs/2509.26580)
2. **SepReformer** — NeurIPS 2024. Base architecture. [github](https://github.com/dmlguq456/SepReformer)
3. **F0-Conditioned U-Net for SATB** — Petermann et al., ISMIR 2020. [arxiv:2008.07645](https://arxiv.org/abs/2008.07645)
4. **BS-RoFormer** — Lu et al., 2023. Band-split transformer, SOTA on MUSDB18. [arxiv:2309.02612](https://arxiv.org/abs/2309.02612)
5. **Mel-RoFormer** — Wang et al., ISMIR 2024. Vocal-specialized transformer. [arxiv:2409.04702](https://arxiv.org/abs/2409.04702)
6. **Vocal Harmony Separation with Time-Domain NNs** — Sarkar et al., Interspeech 2021. PIT for 4-part a cappella.
7. **Conditioned-U-Net** — ISMIR 2019. FiLM conditioning for multi-source separation. [arxiv:1907.01277](https://arxiv.org/abs/1907.01277)
8. **JaCappella Corpus** — [project page](https://tomohikonakamura.github.io/jaCappella_corpus/)
9. **Dagstuhl ChoirSet** — [doi:10.5334/tismir.48](https://transactions.ismir.net/articles/10.5334/tismir.48)

---

## Project Structure

```
music_splitter/
├── Plan.md                  # This file
├── scripts/
│   ├── download_all.sh      # Download all datasets
│   ├── download_jacappella.sh
│   ├── download_dagstuhl.sh
│   ├── download_musdb18hq.sh
│   └── download_acappella.sh
├── data/                    # Downloaded datasets (gitignored)
│   ├── jacappella/
│   ├── dagstuhl_choirset/
│   ├── musdb18hq/
│   └── acappella/
├── src/
│   ├── model/               # SepReformer + SNAKE + FiLM modifications
│   ├── data/                # Dataset loaders + augmentation
│   ├── losses/              # SI-SDR, STFT loss, PIT wrapper
│   └── train.py             # Training entrypoint
├── tests/                   # Unit tests
└── pyproject.toml
```
