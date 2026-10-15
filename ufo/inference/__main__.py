"""Unified inference CLI; task-specific options belong to existing entrypoints."""
import importlib
import sys


def main():
    tasks = ("tracking", "reward", "goal")
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: ./run_inference.sh {tracking,reward,goal} --model-folder RUN [options]\n"
              "Use TASK --help for options. Robot and environment settings come from RUN/config.json.")
        return
    task = sys.argv.pop(1)
    if task not in tasks:
        raise SystemExit(f"Unknown inference task {task!r}; choose tracking, reward or goal.")
    importlib.import_module(f"ufo.inference.{task}_inference").main()


if __name__ == "__main__":
    main()
