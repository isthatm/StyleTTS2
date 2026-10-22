# StyleTTS2 (Kokoro-82M Fine-tuning Fork)

> **Note:** This fork's `main` branch contains the Kokoro-compatible changes from [semidark/StyleTTS2](https://github.com/semidark/StyleTTS2), based on the original [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) repository. It is used as a git submodule for the **[kikiri-tts](https://github.com/isthatm/kikiri-tts)** training recipe.

## Why this fork exists

The [Kokoro-82M](https://github.com/hexgrad/kokoro) TTS model is based on the StyleTTS 2 architecture. However, to fine-tune it using its published HuggingFace weights, several modifications to the upstream training code are required:

1. **PyTorch API Migration (Critical):** Migrated `torch.nn.utils.weight_norm` and `spectral_norm` to the modern `torch.nn.utils.parametrizations` API. This is mandatory for compatibility with Kokoro's inference pipeline (`KModel`).
2. **Kokoro Symbols:** Integrated Kokoro's specific 178-token IPA vocabulary (`kokoro_symbols.py`).
3. **Bug Fixes:**
   - Fixed an `unsqueeze` shape mismatch crash at epoch boundaries involving F0 tensors.
   - Fixed checkpoint saving order to prevent data loss if TensorBoard audio generation fails.
   - Fixed missing `.train()` mode re-initializations after checkpoint loading in Stage 2.
   - Removed hardcoded `ipdb` breakpoints that caused silent hangs.
   - Added a monkey-patch for `torch.load` `weights_only=False` for PyTorch 2.6+ compatibility.
   - Filtered long phoneme sequences (> 510 tokens) to prevent PLBERT position embedding overflows.

## Usage

This repository is not meant to be used standalone. 

Please see the **[kikiri-tts](https://github.com/isthatm/kikiri-tts)** repository for the full end-to-end training guide, dataset preparation scripts, and voicepack extraction tools.

### Resume Stage 2 after an interruption

Keep the saved data lists, architecture, batch size, and loss settings. Add these top-level fields to the training YAML:

```yaml
resume_checkpoint: /absolute/path/to/epoch_2nd_00002.pth
epochs_2nd: 4  # total target; epoch index 2 resumes at epoch 4
```

Run `python train_second.py --config_path /absolute/path/to/full.yml` from this directory.
Recovery restores all model components and optimizer states, preserves the prosodic encoder, and continues epoch and step counters.
Keep `first_stage.pth` for the final voicepack export. Legacy checkpoints omit random-number state, so continuation is not exactly reproducible.

---
*For the original StyleTTS2 project and documentation, please visit the [upstream repository](https://github.com/yl4579/StyleTTS2).*
