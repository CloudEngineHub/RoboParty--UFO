"""Generate and validate a robot draft and optional retargeted-motion manifest."""
import argparse
import json
import shlex
from pathlib import Path

from omegaconf import OmegaConf

from ufo.tools import motion_data, robot_inspect
from ufo.tools.robot_config import load_robot_training_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--xml", type=Path, help="Generate a draft from this MJCF; no automatic retargeting.")
    source.add_argument("--robot-config", type=Path, help="Validate a previously reviewed robot YAML.")
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path, help="New robot YAML path, required with --xml.")
    parser.add_argument("--motion-source", nargs="+", help="Already-retargeted RobotState CSV/NPZ files or glob.")
    parser.add_argument("--format", choices=("auto", "robot_state_csv", "robot_state_npz"), default="auto")
    parser.add_argument("--fps", type=float)
    args = parser.parse_args(argv)
    if args.xml and args.output is None:
        parser.error("--xml requires --output")
    if args.robot_config and (args.output or args.urdf):
        parser.error("--output and --urdf are only used with --xml")
    config = (args.output or args.robot_config).expanduser().resolve()
    manifest = config.with_name(config.stem + "_data.yaml")
    report = config.with_name(config.stem + "_setup.json")
    for path in ([config] if args.xml else []) + ([manifest] if args.motion_source else []) + [report]:
        if path.exists():
            parser.error(f"Refusing to overwrite {path}; choose a new output path or remove an obsolete report explicitly.")
    if args.xml:
        inspect_args = ["--xml", str(args.xml), "--name", args.name, "--out", str(config), "--with-training"]
        if args.urdf:
            inspect_args += ["--urdf", str(args.urdf)]
        robot_inspect.main(inspect_args)
    spec = load_robot_training_spec(config)
    config_data = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    warnings = list(config_data.get("metadata", {}).get("warnings", []))
    warnings.append("Review feet/contact semantics, PD gains, actuator limits, default pose and rewards before training.")
    if args.motion_source:
        build_args = ["--robot", str(config), "--source", *args.motion_source, "--format", args.format,
                      "--name", args.name, "--out", str(manifest)]
        if args.fps is not None:
            build_args += ["--fps", str(args.fps)]
        motion_data.main(["build", *build_args])
    command = None
    if args.motion_source:
        command = shlex.join(["./run_train.sh", "--robot-config", str(config), "--data-manifest", str(manifest),
                              "--agent", "fb", "--gpu-ids", "single", "--smoke", "--work-dir", f"runs/{config.stem}_smoke"])
    result = {
        "robot": spec.robot.name, "robot_config": str(config), "control_joints": spec.robot.control_joint_names,
        "data_manifest": str(manifest) if args.motion_source else None,
        "status": "validated draft; hardware calibration and GPU smoke still required",
        "warnings": warnings,
        "inference_requirements": {"tracking": "matching trained policy and motion data",
                                   "goal": "matching policy, motion data and robot-specific goal JSON",
                                   "reward": "FB policy and replay; non-G1 root/locomotion tasks only"},
        "smoke_command": command,
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Setup report: {report}")
    if command:
        print("After reviewing the draft, run:\n" + command)
    else:
        print("Provide --motion-source with retargeted RobotState data to build a training manifest.")


if __name__ == "__main__":
    main()
