#!/usr/bin/env python3
"""
将当前目录（multi_view_smpl_optimizer/utils）完整复制到指定目标目录。

特点：
- 复制目录结构 + 所有文件（含子目录）
- 同名文件/目录：覆盖（dirs_exist_ok=True + copy2）
- 保留时间戳等元信息（shutil.copy2）
- 尽量“原封不动”：遇到符号链接会按链接本身复制（symlinks=True）
- 安全检查：禁止把目标设为源目录本身，或源目录的子目录（避免递归/自我覆盖）

用法示例：
  python multi_view_smpl_optimizer/utils/copy_utils_dir.py --dest /tmp/utils_copy

  # 如果你想把它复制到某个目录下，并保持目录名为 utils：
  python multi_view_smpl_optimizer/utils/copy_utils_dir.py --into /tmp/target_root
  # 等价于 --dest /tmp/target_root/utils
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


def _realpath(p: Path) -> str:
    # 使用 realpath 规避软链接导致的“看起来不在子目录”的情况
    return os.path.realpath(str(p))


def _is_same_or_inside(child: Path, parent: Path) -> bool:
    child_r = _realpath(child)
    parent_r = _realpath(parent)
    if child_r == parent_r:
        return True
    parent_r_slash = parent_r.rstrip(os.sep) + os.sep
    return child_r.startswith(parent_r_slash)


def _plan_copy_ops(src_dir: Path, dest_dir: Path) -> tuple[list[str], dict[str, int]]:
    """
    返回 (lines, stats)

    - lines: 每条将执行的操作（mkdir/copy/link），用于 dry-run 输出
    - stats: 统计信息（dirs/files/symlinks/overwrites）
    """
    lines: list[str] = []
    stats = {"dirs": 0, "files": 0, "symlinks": 0, "overwrites": 0}

    # root
    if dest_dir.exists():
        lines.append(f"[DIR ] {dest_dir} (exists)")
    else:
        lines.append(f"[MKDIR] {dest_dir}")
    stats["dirs"] += 1

    for root, dirnames, filenames in os.walk(src_dir, topdown=True, followlinks=False):
        root_p = Path(root)
        rel_root = root_p.relative_to(src_dir)
        dest_root = dest_dir / rel_root

        # 不下钻到“源目录里的符号链接目录”里（真实 copytree(symlinks=True) 会复制链接本身）
        pruned: list[str] = []
        for d in dirnames:
            p = root_p / d
            if p.is_symlink():
                pruned.append(d)
        if pruned:
            dirnames[:] = [d for d in dirnames if d not in pruned]
            for d in pruned:
                src_p = root_p / d
                dst_p = dest_root / d
                overwrite = dst_p.exists() or dst_p.is_symlink()
                if overwrite:
                    stats["overwrites"] += 1
                stats["symlinks"] += 1
                lines.append(f"[LINK] {src_p} -> {dst_p}" + (" (overwrite)" if overwrite else ""))

        # directories
        for d in dirnames:
            src_p = root_p / d
            dst_p = dest_root / d
            if dst_p.exists():
                lines.append(f"[DIR ] {dst_p} (exists)")
            else:
                lines.append(f"[MKDIR] {dst_p}")
            stats["dirs"] += 1

        # files (含普通文件 + 符号链接文件)
        for f in filenames:
            src_p = root_p / f
            dst_p = dest_root / f
            overwrite = dst_p.exists() or dst_p.is_symlink()
            if overwrite:
                stats["overwrites"] += 1
            if src_p.is_symlink():
                stats["symlinks"] += 1
                lines.append(f"[LINK] {src_p} -> {dst_p}" + (" (overwrite)" if overwrite else ""))
            else:
                stats["files"] += 1
                lines.append(f"[COPY] {src_p} -> {dst_p}" + (" (overwrite)" if overwrite else ""))

    return lines, stats


def copy_this_dir_to(dest_dir: Path, *, dry_run: bool = False) -> None:
    src_dir = Path(__file__).resolve().parent

    # 目标目录必须是一个目录路径（不存在则创建）
    dest_dir = dest_dir.expanduser()
    if dest_dir.exists() and not dest_dir.is_dir():
        raise ValueError(f"--dest 必须是目录路径，但当前是文件：{dest_dir}")

    # 安全检查：禁止复制到自己或子目录，避免无限递归/破坏源
    if _is_same_or_inside(dest_dir, src_dir):
        raise ValueError(
            "目标目录不能是源目录本身或其子目录。\n"
            f"  src:  {src_dir}\n"
            f"  dest: {dest_dir}"
        )

    if dry_run:
        lines, stats = _plan_copy_ops(src_dir, dest_dir)
        try:
            for line in lines:
                print(line)
            print(
                "[DRY-RUN] summary:"
                f" dirs={stats['dirs']}, files={stats['files']}, symlinks={stats['symlinks']}, overwrites={stats['overwrites']}"
            )
        except BrokenPipeError:
            # 典型场景：输出被 `head`/`less -F` 等提前截断，避免报错污染终端
            return
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 注意：copytree(dirs_exist_ok=True) 会覆盖同名文件，但不会删除目标里多余的文件。
    # 用户需求是“同名注意覆盖”，不要求清理多余内容，所以不做 --delete 行为（更安全）。
    shutil.copytree(
        src_dir,
        dest_dir,
        dirs_exist_ok=True,
        symlinks=True,
        copy_function=shutil.copy2,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="复制当前 utils 目录到目标目录（同名覆盖）。")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dest",
        type=Path,
        help="目标目录（复制后的目录路径就是这个目录，例如 /tmp/utils_copy）。",
    )
    g.add_argument(
        "--into",
        type=Path,
        help="目标根目录（会复制到 <into>/utils 下）。",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的复制/覆盖操作，不做任何实际写入。",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    if ns.into is not None:
        dest = Path(ns.into) / "utils"
    else:
        dest = Path(ns.dest)

    try:
        copy_this_dir_to(dest, dry_run=bool(ns.dry_run))
    except BrokenPipeError:
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if ns.dry_run:
        print(f"[OK] dry-run 完成：{Path(__file__).resolve().parent} -> {dest}")
    else:
        print(f"[OK] 已复制：{Path(__file__).resolve().parent} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

