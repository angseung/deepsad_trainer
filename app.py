"""
Deep SAD — Gradio web UI wrapping train.py and test.py.

Tabs:
  1. Training settings  — configure TrainConfig, export train.yaml, run train.py
  2. Training log       — live stdout stream from train.py
  3. Model validation   — configure TestConfig, export test.yaml, run test.py, show score table
  4. Model test         — pick result dir and display saved FP/FN/TP grid images
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr

from config import TestConfig, TrainConfig

# Project root: directory containing this file.
# All relative paths (yaml, subprocess) are resolved from here so the app
# works correctly regardless of the working directory at launch.
_PROJECT_ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# PyQt6 native file / folder picker helpers
#
# QApplication and all dialogs must run on the Qt main thread.
# Gradio handlers run on worker threads, so we use Qt queued signals to
# dispatch dialog requests to the main thread and block on queue.Queue until
# the result is ready.
#
# _DialogRunner is instantiated in main.py after QApplication is created,
# then assigned to this module's _dialog_runner variable.
# ─────────────────────────────────────────────────────────────────────────────

import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot


class _DialogRunner(QObject):
    """Lives on the Qt main thread; dispatches dialogs via queued signals.

    Signal arguments are plain strings only — passing Python objects (e.g.
    queue.Queue) through QueuedConnection causes PyQt6 reference issues and
    crashes.  Instead, the result is communicated back via a shared string +
    threading.Event.  A lock serialises concurrent dialog requests.
    """

    _req_folder = pyqtSignal(str)
    _req_file = pyqtSignal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._result = ""
        self._req_folder.connect(self._on_folder, Qt.ConnectionType.QueuedConnection)
        self._req_file.connect(self._on_file, Qt.ConnectionType.QueuedConnection)

    @pyqtSlot(str)
    def _on_folder(self, current: str) -> None:
        try:
            from PyQt6.QtWidgets import QFileDialog

            path = QFileDialog.getExistingDirectory(
                None,
                "폴더 선택",
                current or ".",
                QFileDialog.Option.DontUseNativeDialog,
            )
            self._result = path if path else current
        except Exception:
            self._result = current
        finally:
            self._event.set()

    @pyqtSlot(str, str)
    def _on_file(self, current: str, filt: str) -> None:
        try:
            from PyQt6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(
                None,
                "파일 선택",
                current or ".",
                filt,
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            self._result = path if path else current
        except Exception:
            self._result = current
        finally:
            self._event.set()

    def pick_folder(self, current: str = "") -> str:
        with self._lock:
            self._event.clear()
            self._req_folder.emit(current)
            self._event.wait(timeout=120)
            return self._result

    def pick_file(self, current: str = "", filt: str = "All Files (*)") -> str:
        with self._lock:
            self._event.clear()
            self._req_file.emit(current, filt)
            self._event.wait(timeout=120)
            return self._result


# Set by main.py after QApplication is created.
_dialog_runner: _DialogRunner | None = None


def pick_folder(current: str = "") -> str:
    if _dialog_runner is None:
        return current
    return _dialog_runner.pick_folder(current)


def pick_pt_file(current: str = "") -> str:
    if _dialog_runner is None:
        return current
    return _dialog_runner.pick_file(
        current, "PyTorch Model (*.pt *.pth);;All Files (*)"
    )


def pick_image_file(current: str = "") -> str:
    if _dialog_runner is None:
        return current
    return _dialog_runner.pick_file(
        current,
        "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;All Files (*)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared training log state (written by background reader thread)
# ─────────────────────────────────────────────────────────────────────────────

_log_lock = threading.Lock()
_train_log: list[str] = []
_train_proc: subprocess.Popen | None = None


def _append_train_log(line: str) -> None:
    with _log_lock:
        _train_log.append(line)
        if len(_train_log) > 3000:
            _train_log.pop(0)


def _get_train_log() -> str:
    with _log_lock:
        return "\n".join(_train_log)


def _reader_thread(proc: subprocess.Popen) -> None:
    assert proc.stdout
    for line in proc.stdout:
        _append_train_log(line.rstrip("\n"))
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Training actions
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler preview
# ─────────────────────────────────────────────────────────────────────────────


def preview_scheduler(
    scheduler,
    lr,
    n_epochs,
    milestone,
    onecycle_pct,
    onecycle_div,
    onecycle_final,
):
    import tempfile
    from tb_logger import plot_lr_schedule

    cfg = TrainConfig(
        scheduler=str(scheduler),
        lr=float(lr),
        n_epochs=int(n_epochs),
        milestone=int(milestone),
        onecycle_pct_start=float(onecycle_pct),
        onecycle_div_factor=float(onecycle_div),
        onecycle_final_div_factor=float(onecycle_final),
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    fig = plot_lr_schedule(cfg, tmp_path)
    return fig


def start_training(
    data_root,
    test_root,
    proj_dim,
    pretrained,
    freeze_warmup,
    freeze_train,
    seed,
    batch_size,
    num_workers,
    img_size,
    warmup_epochs,
    warmup_lr,
    save_interval,
    scheduler,
    onecycle_pct,
    onecycle_div,
    onecycle_final,
    eta,
    lr,
    n_epochs,
    milestone,
    weight_decay,
    device,
) -> str:
    global _train_proc
    if _train_proc and _train_proc.poll() is None:
        return "⚠ 이미 학습 중입니다."

    cfg = TrainConfig(
        data_root=str(data_root),
        test_root=str(test_root),
        proj_dim=int(proj_dim),
        pretrained=bool(pretrained),
        freeze_backbone_warmup=bool(freeze_warmup),
        freeze_backbone_train=bool(freeze_train),
        seed=int(seed),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        img_size=int(img_size),
        warmup_epochs=int(warmup_epochs),
        warmup_lr=float(warmup_lr),
        save_interval=int(save_interval),
        scheduler=str(scheduler),
        onecycle_pct_start=float(onecycle_pct),
        onecycle_div_factor=float(onecycle_div),
        onecycle_final_div_factor=float(onecycle_final),
        eta=float(eta),
        lr=float(lr),
        n_epochs=int(n_epochs),
        milestone=int(milestone),
        weight_decay=float(weight_decay),
        device=str(device),
    )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        train_yaml_path = f.name
    cfg.to_yaml(train_yaml_path)

    with _log_lock:
        _train_log.clear()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    _train_proc = subprocess.Popen(
        [sys.executable, "train.py", "--config", train_yaml_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=_PROJECT_ROOT,
        env=env,
    )
    threading.Thread(target=_reader_thread, args=(_train_proc,), daemon=True).start()
    return "학습 시작됨. [학습 진행상황] 탭에서 로그를 확인하세요."


def stop_training() -> str:
    if _train_proc and _train_proc.poll() is None:
        _train_proc.terminate()
        return "학습 중단됨."
    return "실행 중인 학습 프로세스가 없습니다."


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard launcher
# ─────────────────────────────────────────────────────────────────────────────

_tb_proc: subprocess.Popen | None = None


def launch_tensorboard(logdir: str) -> str:
    """Start tensorboard if not already running and return an HTML link."""
    global _tb_proc

    # Reuse existing process if still alive
    if _tb_proc and _tb_proc.poll() is None:
        return _tb_link_html(_tb_port)

    port = _tb_port
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    resolved_logdir = str((_PROJECT_ROOT / logdir).resolve())
    _tb_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            resolved_logdir,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=_PROJECT_ROOT,
        env=env,
    )

    # Wait up to 8 s for tensorboard to finish starting
    import time

    deadline = time.monotonic() + 8.0
    assert _tb_proc.stdout
    for line in _tb_proc.stdout:
        if (
            "started" in line.lower()
            or "listening" in line.lower()
            or str(port) in line
        ):
            break
        if time.monotonic() > deadline:
            break

    return _tb_link_html(port)


_tb_port: int = 6006


def _tb_link_html(port: int) -> str:
    url = f"http://127.0.0.1:{port}"
    return (
        f'<a href="{url}" target="_blank" style="font-size:1.1em;">'
        f"TensorBoard 열기 → {url}</a>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation actions
# ─────────────────────────────────────────────────────────────────────────────


def start_validation(
    checkpoint,
    test_root,
    result_dir,
    seed,
    batch_size,
    num_workers,
    img_size,
    num_tp_samples,
    device,
) -> tuple[str, list[list]]:
    cfg = TestConfig(
        checkpoint=str(checkpoint),
        test_root=str(test_root),
        result_dir=str(result_dir),
        seed=int(seed),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        img_size=int(img_size),
        num_tp_samples=int(num_tp_samples),
        device=str(device),
    )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        test_yaml_path = f.name
    cfg.to_yaml(test_yaml_path)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "test.py", "--config", test_yaml_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=_PROJECT_ROOT,
        env=env,
    )
    lines: list[str] = []
    assert proc.stdout
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
    proc.wait()

    table = _parse_validation_output(lines)
    return "\n".join(lines), table


def _parse_validation_output(lines: list[str]) -> list[list]:
    """Extract key metrics from test.py stdout into table rows."""
    rows: list[list] = []
    for line in lines:
        s = line.strip()
        # Score statistics line: "  min=0.0100  max=..."
        if s.startswith("min=") or ("min=" in s and "max=" in s):
            for token in s.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    rows.append([k, v])
        # Threshold line (from checkpoint)
        elif "Threshold" in s and ":" in s:
            label, val = s.split(":", 1)
            rows.append([label.strip(), val.strip()])
        # Sample analysis lines
        elif any(
            s.startswith(prefix)
            for prefix in ("전체 샘플", "정상 샘플", "이상 샘플", "FP ", "FN ", "TP ")
        ):
            if ":" in s:
                label, val = s.split(":", 1)
                rows.append([label.strip(), val.strip()])
        # Best AUC line from training checkpoint
        elif "Best AUC" in s and ":" in s:
            label, val = s.split(":", 1)
            rows.append([label.strip(), val.strip()])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Model test: run inference on arbitrary images and display results
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def run_model_test(
    checkpoint: str,
    image_path: str,
    result_dir: str,
    img_size: int,
    device: str,
) -> str:
    import cv2
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from common import DeepSAD_ResNet50, RealImageDataset
    from config import TrainConfig
    from test import draw_inference_image

    img_p = Path(image_path)
    if img_p.is_file() and img_p.suffix.lower() in _IMG_EXTS:
        file_paths = [str(img_p)]
    elif img_p.is_dir():
        file_paths = sorted(
            str(p) for p in img_p.iterdir() if p.suffix.lower() in _IMG_EXTS
        )
    else:
        return "유효한 이미지 파일 또는 폴더 경로를 입력하세요."

    if not file_paths:
        return "이미지 파일이 없습니다."

    device_t = torch.device(device)
    ckpt = torch.load(str(checkpoint), map_location=device_t, weights_only=True)

    threshold = float(ckpt.get("threshold", 0.0))
    if threshold == 0.0:
        return (
            "체크포인트에 threshold가 없습니다. "
            "해당 모델을 재학습하거나 최신 train.py로 저장된 체크포인트를 사용하세요."
        )

    saved_cfg = TrainConfig(**ckpt.get("cfg", {}))
    c = ckpt["c"].to(device_t)

    model = DeepSAD_ResNet50(
        proj_dim=saved_cfg.proj_dim,
        freeze_backbone=saved_cfg.freeze_backbone_train,
        pretrained=False,
    ).to(device_t)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    normalize = (
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if saved_cfg.pretrained
        else []
    )
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            *normalize,
        ]
    )

    labels_dummy = torch.zeros(len(file_paths), dtype=torch.long)
    loader = DataLoader(
        RealImageDataset(file_paths, labels_dummy, transform),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    all_scores = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device_t)
            z = model(imgs)
            all_scores.append(torch.sum((z - c) ** 2, dim=1).cpu())
    scores = torch.cat(all_scores).numpy()

    rd = Path(result_dir)
    rd.mkdir(parents=True, exist_ok=True)

    for i, (fp, score) in enumerate(zip(file_paths, scores)):
        img = draw_inference_image(fp, float(score), threshold)
        fname = os.path.splitext(os.path.basename(fp))[0]
        label = "ANOMALY" if score > threshold else "NORMAL"
        save_path = str(rd / f"{i:04d}_{label}_{fname}_score{score:.4f}.png")
        cv2.imwrite(save_path, img)

    n_anomaly = int((scores > threshold).sum())
    n_normal = len(scores) - n_anomaly
    return (
        f"완료: {len(file_paths)}개 이미지 처리\n"
        f"  NORMAL: {n_normal}개, ANOMALY: {n_anomaly}개\n"
        f"  Threshold (from checkpoint): {threshold:.4f}\n"
        f"  결과 저장 경로: {result_dir}"
    )


def load_test_images(result_dir: str):
    """Load all result images from result_dir for the test viewer."""
    paths = _collect_images(Path(result_dir))
    img = paths[0] if paths else None
    return img, _counter_text(0, len(paths)), paths, 0


# ─────────────────────────────────────────────────────────────────────────────
# FP / FN image viewer helpers
# ─────────────────────────────────────────────────────────────────────────────


def _collect_images(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(p) for p in directory.iterdir() if p.suffix.lower() == ".png")


def _counter_text(idx: int, total: int) -> str:
    return f"{idx + 1} / {total}" if total else "0 / 0"


def load_fp_fn_images(result_dir: str):
    """Called after validation completes; returns initial state for both viewers."""
    rd = Path(result_dir)
    fp_paths = _collect_images(rd / "FP")
    fn_paths = _collect_images(rd / "FN")
    fp_img = fp_paths[0] if fp_paths else None
    fn_img = fn_paths[0] if fn_paths else None
    return (
        fp_img,
        _counter_text(0, len(fp_paths)),
        fn_img,
        _counter_text(0, len(fn_paths)),
        fp_paths,
        fn_paths,
        0,
        0,
    )


def _navigate(paths: list[str], idx: int, delta: int):
    if not paths:
        return None, "0 / 0", paths, idx
    new_idx = max(0, min(len(paths) - 1, idx + delta))
    return paths[new_idx], _counter_text(new_idx, len(paths)), paths, new_idx


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI builder
# ─────────────────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    cfg_t = TrainConfig()
    cfg_v = TestConfig()

    with gr.Blocks(title="DeepSAD UI") as demo:
        gr.Markdown("# DeepSAD — 이상 탐지 모델 학습 & 검증 UI")

        with gr.Tabs():

            # ── Tab 1: 학습 설정 ──────────────────────────────────────────────
            with gr.TabItem("학습 설정"):
                gr.Markdown("### 데이터 경로")
                with gr.Row():
                    t_data_root = gr.Textbox(
                        label="data_root (학습 데이터 루트)",
                        value=cfg_t.data_root,
                        scale=5,
                    )
                    t_data_root_btn = gr.Button("탐색", scale=1, size="sm")
                with gr.Row():
                    t_test_root = gr.Textbox(
                        label="test_root (검증 데이터 루트)",
                        value=cfg_t.test_root,
                        scale=5,
                    )
                    t_test_root_btn = gr.Button("탐색", scale=1, size="sm")

                gr.Markdown("### 모델 설정")
                t_pretrained = gr.Checkbox(
                    label="pretrained (ImageNet 사전학습 가중치 사용)",
                    value=cfg_t.pretrained,
                )
                with gr.Row(visible=cfg_t.pretrained) as t_freeze_row:
                    t_freeze_warmup = gr.Checkbox(
                        label="freeze_backbone_warmup (워밍업 중 백본 동결)",
                        value=cfg_t.freeze_backbone_warmup,
                    )
                    t_freeze_train = gr.Checkbox(
                        label="freeze_backbone_train (본 학습 중 백본 동결)",
                        value=cfg_t.freeze_backbone_train,
                    )
                t_proj_dim = gr.Number(
                    label="proj_dim (프로젝션 차원)", value=cfg_t.proj_dim, precision=0
                )

                gr.Markdown("### 워밍업 하이퍼파라미터")
                with gr.Row():
                    t_warmup_epochs = gr.Number(
                        label="warmup_epochs", value=cfg_t.warmup_epochs, precision=0
                    )
                    t_warmup_lr = gr.Number(
                        label="warmup_lr (SGD 학습률)", value=cfg_t.warmup_lr
                    )

                gr.Markdown("### 본 학습 하이퍼파라미터")
                with gr.Row():
                    t_seed = gr.Number(label="seed", value=cfg_t.seed, precision=0)
                    t_batch_size = gr.Number(
                        label="batch_size", value=cfg_t.batch_size, precision=0
                    )
                    t_num_workers = gr.Number(
                        label="num_workers", value=cfg_t.num_workers, precision=0
                    )
                    t_img_size = gr.Number(
                        label="img_size", value=cfg_t.img_size, precision=0
                    )
                with gr.Row():
                    t_lr = gr.Number(label="lr (학습률)", value=cfg_t.lr)
                    t_n_epochs = gr.Number(
                        label="n_epochs (총 에포크 수)",
                        value=cfg_t.n_epochs,
                        precision=0,
                    )
                    t_weight_decay = gr.Number(
                        label="weight_decay", value=cfg_t.weight_decay
                    )
                    t_eta = gr.Number(label="eta (레이블 가중치)", value=cfg_t.eta)
                with gr.Row():
                    t_save_interval = gr.Number(
                        label="save_interval (-1: best+last만 저장)",
                        value=cfg_t.save_interval,
                        precision=0,
                    )

                gr.Markdown("### 스케줄러 설정")
                t_scheduler = gr.Dropdown(
                    label="scheduler",
                    choices=["multistep", "onecycle", "combined"],
                    value=cfg_t.scheduler,
                )
                with gr.Row():
                    t_onecycle_pct = gr.Number(
                        label="onecycle_pct_start",
                        value=cfg_t.onecycle_pct_start,
                        interactive=(cfg_t.scheduler != "multistep"),
                    )
                    t_onecycle_div = gr.Number(
                        label="onecycle_div_factor",
                        value=cfg_t.onecycle_div_factor,
                        interactive=(cfg_t.scheduler != "multistep"),
                    )
                    t_onecycle_final = gr.Number(
                        label="onecycle_final_div_factor",
                        value=cfg_t.onecycle_final_div_factor,
                        interactive=(cfg_t.scheduler != "multistep"),
                    )
                t_milestone = gr.Number(
                    label="milestone (MultiStep LR 감소 에포크)",
                    value=cfg_t.milestone,
                    precision=0,
                    interactive=(cfg_t.scheduler != "onecycle"),
                )
                t_device = gr.Textbox(label="device", value=cfg_t.device)

                t_preview_btn = gr.Button(
                    "설정 미리보기 (스케줄러 개형)", variant="secondary"
                )
                t_lr_plot = gr.Plot(label="LR Schedule", visible=False)

                with gr.Row():
                    t_start_btn = gr.Button("학습 시작", variant="primary")
                    t_stop_btn = gr.Button("학습 중단", variant="stop")
                t_status = gr.Textbox(label="상태", interactive=False)

            # ── Tab 2: 학습 진행상황 ──────────────────────────────────────────
            with gr.TabItem("학습 진행상황"):
                log_box = gr.Textbox(
                    label="학습 로그",
                    lines=35,
                    max_lines=50,
                    interactive=False,
                    autoscroll=True,
                )
                with gr.Row():
                    log_refresh_btn = gr.Button("새로고침")
                    with gr.Row():
                        tb_logdir = gr.Textbox(label="logdir", value="out", scale=4)
                        tb_btn = gr.Button("TensorBoard 실행", scale=2)
                tb_link = gr.HTML(visible=False)
                gr.Timer(value=3.0).tick(fn=_get_train_log, outputs=log_box)
                log_refresh_btn.click(fn=_get_train_log, outputs=log_box)

            # ── Tab 3: 모델 검증 ──────────────────────────────────────────────
            with gr.TabItem("모델 검증"):
                gr.Markdown("### 검증 설정")
                with gr.Row():
                    v_checkpoint = gr.Textbox(
                        label="checkpoint (.pt 파일)", value=cfg_v.checkpoint, scale=5
                    )
                    v_checkpoint_btn = gr.Button("탐색", scale=1, size="sm")
                with gr.Row():
                    v_test_root = gr.Textbox(
                        label="test_root (테스트 데이터 루트)",
                        value=cfg_v.test_root,
                        scale=5,
                    )
                    v_test_root_btn = gr.Button("탐색", scale=1, size="sm")
                with gr.Row():
                    v_result_dir = gr.Textbox(
                        label="result_dir (결과 저장 경로)",
                        value=cfg_v.result_dir,
                        scale=5,
                    )
                    v_result_dir_btn = gr.Button("탐색", scale=1, size="sm")

                with gr.Row():
                    v_seed = gr.Number(label="seed", value=cfg_v.seed, precision=0)
                    v_batch_size = gr.Number(
                        label="batch_size", value=cfg_v.batch_size, precision=0
                    )
                    v_num_workers = gr.Number(
                        label="num_workers", value=cfg_v.num_workers, precision=0
                    )
                    v_img_size = gr.Number(
                        label="img_size", value=cfg_v.img_size, precision=0
                    )
                with gr.Row():
                    v_num_tp = gr.Number(
                        label="num_tp_samples", value=cfg_v.num_tp_samples, precision=0
                    )
                    v_device = gr.Textbox(label="device", value=cfg_v.device)

                v_start_btn = gr.Button("검증 시작", variant="primary")
                v_output = gr.Textbox(
                    label="검증 출력 로그", lines=15, interactive=False
                )
                v_table = gr.Dataframe(
                    headers=["항목", "값"],
                    label="결과 요약",
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("### 오분류 샘플 뷰어")
                with gr.Row():
                    # ── FP 블럭 ──────────────────────────────
                    with gr.Column():
                        gr.Markdown("#### FP — 정상 → 이상 오분류")
                        fp_img = gr.Image(
                            label="FP 샘플",
                            type="filepath",
                            interactive=False,
                        )
                        with gr.Row():
                            fp_prev_btn = gr.Button("◀ 이전", size="sm")
                            fp_counter = gr.Textbox(
                                value="0 / 0",
                                show_label=False,
                                interactive=False,
                                scale=1,
                            )
                            fp_next_btn = gr.Button("다음 ▶", size="sm")

                    # ── FN 블럭 ──────────────────────────────
                    with gr.Column():
                        gr.Markdown("#### FN — 이상 → 정상 오분류")
                        fn_img = gr.Image(
                            label="FN 샘플",
                            type="filepath",
                            interactive=False,
                        )
                        with gr.Row():
                            fn_prev_btn = gr.Button("◀ 이전", size="sm")
                            fn_counter = gr.Textbox(
                                value="0 / 0",
                                show_label=False,
                                interactive=False,
                                scale=1,
                            )
                            fn_next_btn = gr.Button("다음 ▶", size="sm")

                fp_paths_state = gr.State([])
                fn_paths_state = gr.State([])
                fp_idx_state = gr.State(0)
                fn_idx_state = gr.State(0)

            # ── Tab 4: 모델 테스트 ────────────────────────────────────────────
            with gr.TabItem("모델 테스트"):
                gr.Markdown("### 입력 설정")
                with gr.Row():
                    mt_checkpoint = gr.Textbox(
                        label="모델 체크포인트 (.pt)", value=cfg_v.checkpoint, scale=5
                    )
                    mt_checkpoint_btn = gr.Button("탐색", scale=1, size="sm")
                with gr.Row():
                    mt_image_path = gr.Textbox(
                        label="이미지 / 이미지 폴더 경로", value="", scale=4
                    )
                    mt_img_file_btn = gr.Button("파일", scale=1, size="sm")
                    mt_img_folder_btn = gr.Button("폴더", scale=1, size="sm")
                with gr.Row():
                    mt_result_dir = gr.Textbox(
                        label="결과 저장 경로", value="out/result_test", scale=5
                    )
                    mt_result_dir_btn = gr.Button("탐색", scale=1, size="sm")

                gr.Markdown("### 추론 옵션")
                with gr.Row():
                    mt_img_size = gr.Number(
                        label="img_size", value=cfg_v.img_size, precision=0
                    )
                    mt_device = gr.Textbox(label="device", value=cfg_v.device)

                mt_start_btn = gr.Button("테스트 시작", variant="primary")
                mt_status = gr.Textbox(label="상태", interactive=False, lines=4)

                gr.Markdown("### 결과 이미지")
                mt_img = gr.Image(
                    label="결과 이미지", type="filepath", interactive=False
                )
                with gr.Row():
                    mt_prev_btn = gr.Button("◀ 이전", size="sm")
                    mt_counter = gr.Textbox(
                        value="0 / 0",
                        show_label=False,
                        interactive=False,
                        scale=1,
                    )
                    mt_next_btn = gr.Button("다음 ▶", size="sm")

                mt_paths_state = gr.State([])
                mt_idx_state = gr.State(0)

        # ── Event wiring ─────────────────────────────────────────────────────

        # Tab 2: TensorBoard
        def _launch_tb(logdir: str):
            html = launch_tensorboard(logdir)
            return gr.update(value=html, visible=True)

        tb_btn.click(fn=_launch_tb, inputs=tb_logdir, outputs=tb_link)

        # Tab 1: scheduler preview
        def _preview(scheduler, lr, n_epochs, milestone, pct, div, final_div):
            fig = preview_scheduler(
                scheduler, lr, n_epochs, milestone, pct, div, final_div
            )
            return gr.update(value=fig, visible=True)

        t_preview_btn.click(
            fn=_preview,
            inputs=[
                t_scheduler,
                t_lr,
                t_n_epochs,
                t_milestone,
                t_onecycle_pct,
                t_onecycle_div,
                t_onecycle_final,
            ],
            outputs=t_lr_plot,
        )

        # Tab 1: path pickers
        t_data_root_btn.click(fn=pick_folder, inputs=t_data_root, outputs=t_data_root)
        t_test_root_btn.click(fn=pick_folder, inputs=t_test_root, outputs=t_test_root)

        # Tab 1: pretrained toggle — enable/disable freeze checkboxes
        def _on_pretrained(val: bool):
            return gr.update(visible=val)

        t_pretrained.change(_on_pretrained, t_pretrained, t_freeze_row)

        # Tab 1: scheduler toggle — enable/disable onecycle params and milestone
        def _on_scheduler(val: str):
            oc = val in ("onecycle", "combined")
            ms = val in ("multistep", "combined")
            return (
                gr.update(interactive=oc),
                gr.update(interactive=oc),
                gr.update(interactive=oc),
                gr.update(interactive=ms),
            )

        t_scheduler.change(
            _on_scheduler,
            t_scheduler,
            [t_onecycle_pct, t_onecycle_div, t_onecycle_final, t_milestone],
        )

        # Tab 1: start / stop
        t_start_btn.click(
            fn=start_training,
            inputs=[
                t_data_root,
                t_test_root,
                t_proj_dim,
                t_pretrained,
                t_freeze_warmup,
                t_freeze_train,
                t_seed,
                t_batch_size,
                t_num_workers,
                t_img_size,
                t_warmup_epochs,
                t_warmup_lr,
                t_save_interval,
                t_scheduler,
                t_onecycle_pct,
                t_onecycle_div,
                t_onecycle_final,
                t_eta,
                t_lr,
                t_n_epochs,
                t_milestone,
                t_weight_decay,
                t_device,
            ],
            outputs=t_status,
        )
        t_stop_btn.click(fn=stop_training, outputs=t_status)

        # Tab 3: path pickers
        v_checkpoint_btn.click(
            fn=pick_pt_file, inputs=v_checkpoint, outputs=v_checkpoint
        )
        v_test_root_btn.click(fn=pick_folder, inputs=v_test_root, outputs=v_test_root)
        v_result_dir_btn.click(
            fn=pick_folder, inputs=v_result_dir, outputs=v_result_dir
        )

        # Tab 3: start validation → then load FP/FN images
        _viewer_outputs = [
            fp_img,
            fp_counter,
            fn_img,
            fn_counter,
            fp_paths_state,
            fn_paths_state,
            fp_idx_state,
            fn_idx_state,
        ]
        v_start_btn.click(
            fn=start_validation,
            inputs=[
                v_checkpoint,
                v_test_root,
                v_result_dir,
                v_seed,
                v_batch_size,
                v_num_workers,
                v_img_size,
                v_num_tp,
                v_device,
            ],
            outputs=[v_output, v_table],
        ).then(fn=load_fp_fn_images, inputs=v_result_dir, outputs=_viewer_outputs)

        # Tab 3: FP navigation
        fp_prev_btn.click(
            fn=lambda paths, idx: _navigate(paths, idx, -1),
            inputs=[fp_paths_state, fp_idx_state],
            outputs=[fp_img, fp_counter, fp_paths_state, fp_idx_state],
        )
        fp_next_btn.click(
            fn=lambda paths, idx: _navigate(paths, idx, +1),
            inputs=[fp_paths_state, fp_idx_state],
            outputs=[fp_img, fp_counter, fp_paths_state, fp_idx_state],
        )

        # Tab 3: FN navigation
        fn_prev_btn.click(
            fn=lambda paths, idx: _navigate(paths, idx, -1),
            inputs=[fn_paths_state, fn_idx_state],
            outputs=[fn_img, fn_counter, fn_paths_state, fn_idx_state],
        )
        fn_next_btn.click(
            fn=lambda paths, idx: _navigate(paths, idx, +1),
            inputs=[fn_paths_state, fn_idx_state],
            outputs=[fn_img, fn_counter, fn_paths_state, fn_idx_state],
        )

        # Tab 4: path pickers
        mt_checkpoint_btn.click(
            fn=pick_pt_file, inputs=mt_checkpoint, outputs=mt_checkpoint
        )
        mt_img_file_btn.click(
            fn=pick_image_file, inputs=mt_image_path, outputs=mt_image_path
        )
        mt_img_folder_btn.click(
            fn=pick_folder, inputs=mt_image_path, outputs=mt_image_path
        )
        mt_result_dir_btn.click(
            fn=pick_folder, inputs=mt_result_dir, outputs=mt_result_dir
        )

        # Tab 4: run test → load results into viewer
        _mt_viewer_outputs = [mt_img, mt_counter, mt_paths_state, mt_idx_state]
        mt_start_btn.click(
            fn=run_model_test,
            inputs=[
                mt_checkpoint,
                mt_image_path,
                mt_result_dir,
                mt_img_size,
                mt_device,
            ],
            outputs=mt_status,
        ).then(fn=load_test_images, inputs=mt_result_dir, outputs=_mt_viewer_outputs)

        # Tab 4: navigation
        mt_prev_btn.click(
            fn=lambda paths, idx: _navigate(paths, idx, -1),
            inputs=[mt_paths_state, mt_idx_state],
            outputs=[mt_img, mt_counter, mt_paths_state, mt_idx_state],
        )
        mt_next_btn.click(
            fn=lambda paths, idx: _navigate(paths, idx, +1),
            inputs=[mt_paths_state, mt_idx_state],
            outputs=[mt_img, mt_counter, mt_paths_state, mt_idx_state],
        )

    return demo
