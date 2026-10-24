# Member 4: PPO clipping sensitivity and shared visualization

## Status

Code and report are provided. Full training has **not** run: PyTorch, Gymnasium and pygame were unavailable and network installation failed. There are no measured PPO results, trained PPO checkpoints or generated gameplay GIFs in this package. Five numerical GAE tests, Python syntax checks, notebook schema validation and sequential execution of the default (training-disabled) notebook cells passed. Jupyter kernel execution was blocked by a local socket permission error. Training and end-to-end runtime compatibility remain unverified. See `validation_status.json`.

## Files

- `member4_section4.md`: English assignment section, with explicit experiment status.
- `Member4_PPO.ipynb`: English notebook; training disabled by default.
- `Project_and_Member4_Guide_ZH.md`: detailed Chinese explanation and integration instructions.
- `member4_experiment.py`: fixed-budget, five-seed, three-clip experiment; separate smoke/final outputs.
- `advantages.py`, `test_advantages.py`: GAE and numerical tests.
- `make_figures.py`: real-result tables and plots, only after training.
- `visualize_progress.py`: PPO/AC/DQN checkpoint comparison GIF renderer.
- `environment.py`, `ppo.py`, `model.py`, `dqn.py`: unchanged upstream dependencies.
- `source_provenance.json`: hashes of upstream module copies.
- `presentation_script_EN.md`: theory/methods narration, with results still pending.

## Run (Python 3.11 or 3.12 recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest test_advantages -v
python member4_experiment.py --mode smoke
python member4_experiment.py --mode final
```

Run inside this folder. Windows activation: `.venv\Scripts\activate`.
Successful final execution produces 15 model runs, selected and staged checkpoints, per-episode test records, `observed_results.md`, learning curves, and a five-stage progress GIF. Keep all outputs with the notebook. Add observed findings to Section 4.7 and update the status only after verifying execution. Do not present smoke results as final findings.

```bash
python member4_experiment.py --mode analysis
python visualize_progress.py --checkpoints early.pt late.pt --labels Early Late --output progress.gif
```

The notebook can be merged into Section 4 of the group notebook. The current experiment chooses the allowed clipping-sensitivity option, not an unmatched comparison against Member 2's AC results. No original project files were edited.
