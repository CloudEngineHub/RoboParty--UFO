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

UFO 是基于 **mjlab** 的人形机器人无监督强化学习框架，支持 **Unitree G1 / RoboParty RP1.3**、**FB-CPR-Aux / TeCH**、多卡训练，以及 tracking、reward、goal 三种推理。

**RP1 的 TeCH 策略尚未进行真机测试。**

**G1 和 RP1 的最终训练数据也将开源，发布后会在这里补充下载入口。**

## 安装

需要 Linux 和兼容 CUDA 12.8 的 NVIDIA 驱动。锁文件固定 PyTorch 2.7.1 / CUDA 12.8 和 NCCL 2.26.5，支持 Blackwell 显卡。执行以下命令，后续操作均在仓库根目录进行：

```bash
git clone https://github.com/Roboparty/UFO.git
cd UFO
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
```

本文适用于包含 `ufo/` 和 `run_inference.sh` 的 mjlab 版本，不适用于旧的 `humanoidverse/` 版本。保存视频需要 `ffmpeg`，Ubuntu 可用 `sudo apt install ffmpeg` 安装。

## 准备数据

现有 G1 LaFAN 训练数据可通过以下命令下载：

```bash
bash scripts/download_data.sh g1_lafan
```

RP1 暂需自行准备原始 LaFAN v7 NPZ 文件，按 [RP1 数据导入说明](docs/ROBOT_ADAPTATION.md#rp1-data-import) 转换。每个机器人需要匹配的动作数据，G1 数据不能直接用于 RP1。

已有转换后的数据或无法访问 Hugging Face 时，见[本地数据放置说明](docs/ROBOT_ADAPTATION.md#use-converted-data)。RP1 数据准备完成后才能启动训练。

## 训练

准备好数据后，选择一条命令运行：

```bash
./run_train.sh --robot g1  --agent fb   --gpu-ids all
./run_train.sh --robot rp1 --agent fb   --gpu-ids all
./run_train.sh --robot g1  --agent tech --gpu-ids all
./run_train.sh --robot rp1 --agent tech --gpu-ids all
```

`--gpu-ids all` 使用所有可见 GPU，通过 `CUDA_VISIBLE_DEVICES` 选择显卡。`--gpu-ids single --smoke` 只检查启动和采样，不更新权重；需要验证参数更新时，使用[短训练检查](docs/TRAIN_INFERENCE.md#short-training-check)。

默认输出到 `runs/<robot>_<agent>`。每个新实验请使用独立的 `--work-dir`；目录中已有 checkpoint 时会恢复训练。启用 W&B：

```bash
uv run wandb login
WANDB_PROJECT=UFO ./run_train.sh --robot g1 --agent fb --gpu-ids all \
  --work-dir runs/g1_fb_exp --use-wandb --wandb-run-name g1_fb_exp
```

## 推理

推理自动读取训练时保存的机器人配置。使用对应机器人的 checkpoint 和匹配的动作数据：

```bash
./run_inference.sh tracking --model-folder runs/g1_fb --motion-list 0 --save-mp4
./run_inference.sh reward --model-folder runs/g1_fb --save-mp4
./run_inference.sh goal --model-folder runs/g1_fb \
  --data-path ufo/data/lafan_29dof.pkl --save-mp4
```

- **Tracking**：FB、TeCH 均支持 G1 和 RP1；可用 `--data-path` 指定动作数据，视频和 z 保存到实验目录。
- **Reward**：仅支持 FB，需要保存的 rank-0 replay；RP1 当前支持根节点运动与移动类任务。
- **Goal**：FB、TeCH 均支持 G1 和 RP1；内置 G1 目标对应完整动作文件。RP1 命令和目标 JSON 示例见 [Goal 推理](docs/TRAIN_INFERENCE.md#goal-inference)。

完整参数见 `./run_inference.sh <tracking|reward|goal> --help`。

## 更多文档

- [训练与推理](docs/TRAIN_INFERENCE.md)：默认配置、评估、checkpoint 和推理参数。
- [机器人与数据适配](docs/ROBOT_ADAPTATION.md)：通过 `robot_setup` 添加机器人、导入已重定向的动作数据。
- [代码结构](docs/ARCHITECTURE.md)：目录与模块职责。
- [真机部署](https://github.com/Roboparty/UFO/tree/deploy)。

## 引用与许可

<details>
<summary>引用 UFO</summary>

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

UFO 采用 [CC BY-NC 4.0](LICENSE) 许可，用于非商业研究。
