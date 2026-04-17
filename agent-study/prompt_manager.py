import os
import yaml
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def list_versions():
    """列出所有可用的 prompt 版本"""
    versions = []
    for f in sorted(PROMPTS_DIR.glob("v*.yaml")):
        with open(f, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        versions.append({
            "version": data["version"],
            "description": data.get("description", ""),
            "file": f.name,
        })
    return versions


def load_prompt(version="v1"):
    """加载指定版本的 system_prompt"""
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt 版本 {version} 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"]


def load_latest_prompt():
    """加载最新版本的 system_prompt（按文件名排序取最后一个）"""
    files = sorted(PROMPTS_DIR.glob("v*.yaml"))
    if not files:
        raise FileNotFoundError(f"没有找到任何 prompt 版本文件: {PROMPTS_DIR}")
    with open(files[-1], "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"]
