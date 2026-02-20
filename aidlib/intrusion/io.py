from __future__ import annotations

import json
import logging
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class IOContext:
    stage: str
    run_ts: str
    out_base: str
    out_dir: Path
    log_dir: Path
    cmd_path: Path
    log_path: Path
    video_path: Path
    scores_path: Path
    meta_path: Path
    params_path: Path


def now_run_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path, level: str = "INFO") -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    return root


def init_io(
    *,
    stage: str,
    out_root: str | Path,
    log_root: str | Path,
    out_base: str,
    run_ts: str,
    argv: Iterable[str],
    log_level: str = "INFO",
) -> IOContext:
    run_ts_eff = str(run_ts).strip() or now_run_ts()
    out_base_eff = str(out_base).strip() or "run"

    out_root_p = Path(out_root)
    log_root_p = Path(log_root)
    out_dir = out_root_p / stage / run_ts_eff / out_base_eff
    log_dir = log_root_p / stage

    _ensure_dir(out_dir)
    _ensure_dir(log_dir)

    cmd_path = log_dir / f"{out_base_eff}_{run_ts_eff}.cmd.txt"
    log_path = log_dir / f"{out_base_eff}_{run_ts_eff}.log"

    cmd_line = " ".join(shlex.quote(str(x)) for x in argv)
    cmd_path.write_text(cmd_line + "\n", encoding="utf-8")
    setup_logger(log_path=log_path, level=log_level)

    return IOContext(
        stage=stage,
        run_ts=run_ts_eff,
        out_base=out_base_eff,
        out_dir=out_dir,
        log_dir=log_dir,
        cmd_path=cmd_path,
        log_path=log_path,
        video_path=out_dir / f"{out_base_eff}_mvp.mp4",
        scores_path=out_dir / "scores.jsonl",
        meta_path=out_dir / "meta.json",
        params_path=out_dir / "params_used.yaml",
    )


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    chunk: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in text:
        if quote is not None:
            chunk.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            chunk.append(ch)
            continue
        if ch in "{[(":
            depth += 1
            chunk.append(ch)
            continue
        if ch in "}])":
            depth = max(0, depth - 1)
            chunk.append(ch)
            continue
        if ch == sep and depth == 0:
            token = "".join(chunk).strip()
            if token:
                parts.append(token)
            chunk = []
            continue
        chunk.append(ch)
    token = "".join(chunk).strip()
    if token:
        parts.append(token)
    return parts


def _parse_scalar(token: str) -> Any:
    v = token.strip()
    if not v:
        return ""
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    lo = v.lower()
    if lo in {"true", "yes", "on"}:
        return True
    if lo in {"false", "no", "off"}:
        return False
    if lo in {"null", "none", "~"}:
        return None
    try:
        if any(c in v for c in (".", "e", "E")):
            return float(v)
        return int(v)
    except ValueError:
        return v


def _parse_inline_map(token: str) -> dict[str, Any]:
    inner = token.strip()[1:-1].strip()
    out: dict[str, Any] = {}
    if not inner:
        return out
    for part in _split_top_level(inner, ","):
        if ":" not in part:
            raise ValueError(f"Invalid inline map item: '{part}'")
        k, v = part.split(":", 1)
        out[k.strip()] = _parse_scalar(v.strip())
    return out


def _parse_value(token: str) -> Any:
    t = token.strip()
    if t.startswith("{") and t.endswith("}"):
        return _parse_inline_map(t)
    return _parse_scalar(t)


def _load_yaml_minimal(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw in text.splitlines():
        no_comment = raw.split("#", 1)[0].rstrip()
        if not no_comment.strip():
            continue
        indent = len(no_comment) - len(no_comment.lstrip(" "))
        content = no_comment.strip()
        if ":" not in content:
            raise ValueError(f"Invalid YAML line: '{raw}'")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid YAML key in line: '{raw}'")

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_value(value)
    return root


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: '{cfg_path}'")
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError("YAML root must be a mapping")
        return parsed
    except ModuleNotFoundError:
        parsed = _load_yaml_minimal(text)
        if not isinstance(parsed, dict):
            raise ValueError("Config root must be a mapping")
        return parsed


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v), ensure_ascii=False)


def _dump_yaml_minimal(data: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(data, Mapping):
        for k, v in data.items():
            if isinstance(v, Mapping):
                lines.append(f"{pad}{k}:")
                lines.extend(_dump_yaml_minimal(v, indent + 2))
            elif isinstance(v, list):
                lines.append(f"{pad}{k}:")
                if not v:
                    lines.append(f"{pad}  []")
                else:
                    for item in v:
                        if isinstance(item, Mapping):
                            lines.append(f"{pad}  -")
                            lines.extend(_dump_yaml_minimal(item, indent + 4))
                        else:
                            lines.append(f"{pad}  - {_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return lines
    if isinstance(data, list):
        for item in data:
            lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(data)}"]


def save_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    dst = Path(path)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
        dst.write_text(text, encoding="utf-8")
    except ModuleNotFoundError:
        lines = _dump_yaml_minimal(dict(payload), indent=0)
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(fp, row: Mapping[str, Any]) -> None:
    fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def create_video_writer(cv2, path: str | Path, width: int, height: int, fps: float, codec: str = "mp4v"):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: '{path}'")
    return writer

