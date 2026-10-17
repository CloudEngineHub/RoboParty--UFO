<h1 align="center">
  UFO: A General <b>U</b>nsupervised Reinforcement Learning <b>F</b>ramework for Humanoid C<b>O</b>ntrol
</h1>

<p align="center">
  <a href="https://roboparty.github.io/UFO/"><img alt="Website" src="https://img.shields.io/badge/Website-roboparty.github.io%2FUFO-2563eb?style=for-the-badge" /></a>
  <a href="https://youtu.be/uJPcLdn9sNA"><img alt="Video" src="https://img.shields.io/badge/Video-Demo-7c3aed?style=for-the-badge" /></a>
  <a href="https://roboparty.github.io/UFO/assets/UFO.pdf"><img alt="PDF" src="https://img.shields.io/badge/PDF-Available-0f766e?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="./assets/UFO.png" alt="UFO" height="72" style="vertical-align: middle;" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./assets/rplab_logo.png" alt="ROBO PARTY LAB Logo" height="72" style="vertical-align: middle;" />
</p>

UFO is an unsupervised reinforcement learning framework for humanoid control, built on **mjlab**. It supports **Unitree G1 / RoboParty RP1.3**, **FB-CPR-Aux / TeCH**, multi-GPU training, and tracking, reward and goal inference.

**The RP1 TeCH policy has not been tested on the physical robot.**

**[G1 / RP1 training data](https://huggingface.co/datasets/xuewang/UFO-MotionData) and the [G1 FB policy tested on hardware](https://huggingface.co/xuewang/ufo-g1-policy) are available.**

## News

**2026-10-15 — Unified mjlab release**

- **Simpler framework, two robots:** replaced the legacy `humanoidverse` layout with `ufo`, with unified G1 / RP1 and FB / TeCH training commands.
- **More efficient training:** evaluation across all GPUs, compiled network forwards, fused Adam, bucketed gradient synchronization, and separate model/replay save schedules.
- **Updated FB defaults:** 1024-wide networks with a shared critic trunk; independent critics remain available via `--no-shared-critic`.
- **Correctness fixes:** preserve true final observations across resets, correct contact-force direction, and align reference and simulator observation scaling.
- **Deployment exports:** ONNX exports include joint order, observation scales, history order, motor gains and action scales. RP1 training data and a G1 FB policy tested on hardware are now available through the links above.

See [architecture](docs/ARCHITECTURE.md) and [training/inference](docs/TRAIN_INFERENCE.md) for details.

## Install

Requires Linux and an NVIDIA driver compatible with CUDA 12.8. The lockfile selects PyTorch 2.7.1 / CUDA 12.8 and NCCL 2.26.5, including Blackwell support. Run these commands, then work from the repository root:

```bash
git clone https://github.com/Roboparty/UFO.git
cd UFO
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
```

This guide requires the mjlab checkout containing `ufo/` and `run_inference.sh`; it does not apply to the older `humanoidverse/` release. Video export requires `ffmpeg` (`sudo apt install ffmpeg` on Ubuntu).

## Prepare data

Download matching full motions and training clips; the script verifies SHA256:

```bash
bash scripts/download_data.sh g1_lafan
bash scripts/download_data.sh rp1_lafan
```

G1 contains 40 motions. RP1 contains 55 motions at 50 Hz, converted for its 24 controlled joints, with 1,166 training clips. Each robot requires matching data; G1 data cannot be used directly for RP1.

For local files, see [data paths](docs/ROBOT_ADAPTATION.md#use-converted-data). To convert your own RP1 NPZ files, see [RP1 data import](docs/ROBOT_ADAPTATION.md#rp1-data-import).

## Train

After preparing data, choose one command:

```bash
./run_train.sh --robot g1  --agent fb   --gpu-ids all
./run_train.sh --robot rp1 --agent fb   --gpu-ids all
./run_train.sh --robot g1  --agent tech --gpu-ids all
./run_train.sh --robot rp1 --agent tech --gpu-ids all
```

`--gpu-ids all` uses all visible GPUs; use `CUDA_VISIBLE_DEVICES` to select them. `--gpu-ids single --smoke` checks startup and sampling only; it does not update weights. See [a short training check with updates](docs/TRAIN_INFERENCE.md#short-training-check).

Runs default to `runs/<robot>_<agent>`. Use a new `--work-dir` for each experiment; an existing checkpoint in that directory is resumed. To enable W&B:

```bash
uv run wandb login
WANDB_PROJECT=UFO ./run_train.sh --robot g1 --agent fb --gpu-ids all \
  --work-dir runs/g1_fb_exp --use-wandb --wandb-run-name g1_fb_exp
```

## Inference

Inference reads robot settings from the saved run. Use a checkpoint trained for the selected robot and matching motion data:

```bash
./run_inference.sh tracking --model-folder runs/g1_fb --motion-list 0 --save-mp4
./run_inference.sh reward --model-folder runs/g1_fb --save-mp4
./run_inference.sh goal --model-folder runs/g1_fb \
  --data-path ufo/data/lafan_29dof.pkl --save-mp4
```

- **Tracking:** FB and TeCH, G1 and RP1. Add `--data-path` to select motion data; videos and z are saved under the run directory.
- **Reward:** FB only; requires saved rank-0 replay. RP1 currently supports root/locomotion tasks.
- **Goal:** FB and TeCH, G1 and RP1. Use full-motion data for the bundled G1 goals. For RP1 commands and a goal JSON example, see [goal inference](docs/TRAIN_INFERENCE.md#goal-inference).

See `./run_inference.sh <tracking|reward|goal> --help` for options.

The [released G1 FB model](https://huggingface.co/xuewang/ufo-g1-policy) includes the approximately 192M-step checkpoint, ONNX exports and matching tracking, goal and forward-walking z. Download it into `model/`:

```bash
uv run hf download xuewang/ufo-g1-policy --local-dir model
./run_inference.sh tracking --model-folder model/g1_policy \
  --data-path ufo/data/lafan_29dof.pkl --motion-list 7 --save-mp4
```

The download includes model weights, but no optimizer state or replay buffer. Computing new reward z requires your own saved replay; the provided reward z is ready for deployment.

## Documentation

- [Training and inference](docs/TRAIN_INFERENCE.md): defaults, evaluation, checkpoints and inference options.
- [Robot and data adaptation](docs/ROBOT_ADAPTATION.md): add robots with `robot_setup` and import retargeted motions.
- [Architecture](docs/ARCHITECTURE.md): code layout and component responsibilities.
- [G1 real-robot deployment](https://github.com/Roboparty/UFO/tree/deploy).
- **RP1 UFO deployment:** [Roboparty/roboparty_deploy](https://github.com/Roboparty/roboparty_deploy).

## Citation and license

<details>
<summary>Cite UFO</summary>

```bibtex
@misc{ufo2026,
  author       = {{RoboParty Lab Team}},
  title        = {UFO: An Unsupervised Reinforcement Learning Framework for Humanoid Control},
  year         = {2026},
  howpublished = {\url{https://github.com/Roboparty/UFO}},
  note         = {Project page: \url{https://roboparty.github.io/UFO/}}
}
```

</details>

UFO is released under [CC BY-NC 4.0](LICENSE) for non-commercial research use.
