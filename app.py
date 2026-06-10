"""
AIDC Tech Video Processor — เว็บ UI ประมวลผลไฟล์วิดีโอด้วย Ultralytics YOLO
รัน:  D:\\yoloe\\venv\\Scripts\\python.exe D:\\yoloe\\app.py
แล้วเปิดเบราว์เซอร์ตามลิงก์ที่ขึ้น (ปกติ http://127.0.0.1:7860)
"""

import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd
import torch
from ultralytics import YOLO, YOLOWorld

# โมเดล open-vocabulary สำหรับตรวจจับวัตถุ "ตามคำที่พิมพ์" (เช่น tree)
WORLD_MODEL = "yolov8s-world.pt"

# รายการวัตถุสำเร็จรูป: ป้ายภาษาไทย (แสดงในตาราง) → คำสั่งภาษาอังกฤษ (ส่งให้ AI)
PRESETS = {
    "👤 คน": "person",
    "🐾 สัตว์": "animal",
    "🐶 สุนัข": "dog",
    "🐦 นก": "bird",
    "🌳 ต้นไม้": "tree",
    "🪵 ท่อนไม้ใหญ่": "wood log",
    "🌿 พุ่มไม้": "bush",
    "🌊 ลำธาร/แม่น้ำ": "river",
    "🔥 กองไฟ/ไฟ": "fire",
    "💨 ควันไฟ": "smoke",
    "⛰️ ภูเขา": "mountain",
    "🪨 ก้อนหิน": "rock",
    "🚗 รถยนต์": "car",
    "🏠 อาคาร/สิ่งปลูกสร้าง": "building",
    "🛣️ ถนน/ทางเดิน": "road",
    "🚧 ป้าย/เครื่องหมาย": "sign",
}
# แผนที่ย้อนกลับ: คำอังกฤษ → ป้ายไทย (ไว้แปลชื่อในตารางผลลัพธ์)
PROMPT_TO_TH = {v: k for k, v in PRESETS.items()}

# ชุดสำเร็จรูป (ปุ่มกดเลือกทีเดียวหลายอย่าง)
QUICK_SETS = {
    "🌲 ป่าไม้/ธรรมชาติ": ["🌳 ต้นไม้", "🪵 ท่อนไม้ใหญ่", "🌿 พุ่มไม้",
                          "🌊 ลำธาร/แม่น้ำ", "⛰️ ภูเขา", "🪨 ก้อนหิน"],
    "🔥 ไฟป่า/ควัน": ["🔥 กองไฟ/ไฟ", "💨 ควันไฟ", "🌳 ต้นไม้", "👤 คน"],
    "👥 คน/สัตว์": ["👤 คน", "🐾 สัตว์", "🐶 สุนัข", "🐦 นก"],
}

ROOT = Path(r"D:\yoloe")
# ทำงานในโฟลเดอร์ D:\yoloe เสมอ เพื่อให้โมเดลที่ดาวน์โหลดอัตโนมัติ (เช่น
# yolov8s-world.pt, CLIP) ลงที่นี่ ไม่ไปเลอะโฟลเดอร์อื่นที่สั่งรันแอป
os.chdir(ROOT)
HAS_CUDA = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU only"

# แคชโมเดลที่โหลดแล้ว เพื่อไม่ต้องโหลดซ้ำทุกครั้ง
_MODEL_CACHE: dict[str, YOLO] = {}


def find_models() -> list[str]:
    """หาไฟล์ .pt ทั้งหมดในโฟลเดอร์ D:\\yoloe"""
    models = sorted(str(p) for p in ROOT.glob("*.pt"))
    return models or ["yolo26n.pt"]


def get_model(model_path: str) -> YOLO:
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def get_world_model(classes: list[str]) -> YOLOWorld:
    """โมเดล open-vocabulary — ตั้งคลาสตามคำที่ผู้ใช้พิมพ์ (เช่น tree, dog)"""
    if WORLD_MODEL not in _MODEL_CACHE:
        _MODEL_CACHE[WORLD_MODEL] = YOLOWorld(WORLD_MODEL)
    model = _MODEL_CACHE[WORLD_MODEL]
    model.set_classes(classes)
    return model


def parse_classes(text: str) -> list[str]:
    """แยกคำที่คั่นด้วย , หรือขึ้นบรรทัดใหม่ → ลิสต์คลาส"""
    return [c.strip() for c in (text or "").replace("\n", ",").split(",") if c.strip()]


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _reencode_h264(src: str) -> str:
    """แปลงเป็น H.264 ให้เบราว์เซอร์เล่นได้ลื่น (ถ้ามี ffmpeg)"""
    if not _has_ffmpeg():
        return src
    dst = src.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", dst],
            check=True, capture_output=True,
        )
        return dst
    except Exception:
        return src


def process_video(video_path, model_path, preset_labels, custom_classes, conf, use_gpu,
                  frame_stride, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("กรุณาอัปโหลดไฟล์วิดีโอก่อน")

    device = 0 if (use_gpu and HAS_CUDA) else "cpu"
    # รวมรายการที่ติ๊กเลือก + ที่พิมพ์เอง → ลิสต์คำสั่ง (ตัดซ้ำ คงลำดับ)
    chosen = [PRESETS[l] for l in (preset_labels or []) if l in PRESETS]
    chosen = list(dict.fromkeys(chosen + parse_classes(custom_classes)))
    if chosen:
        progress(0, desc=f"กำลังโหลดโมเดล AI สำหรับ: {', '.join(chosen)}")
        model = get_world_model(chosen)   # ตรวจจับตามรายการที่เลือก/พิมพ์
        mode_label = "AI ตรวจจับเอง (" + ", ".join(chosen) + ")"
    else:
        model = get_model(model_path)     # โมเดลปกติ (80 คลาส COCO)
        mode_label = Path(model_path).name
    stride = max(1, int(frame_stride))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("เปิดไฟล์วิดีโอไม่ได้ — ไฟล์อาจเสียหรือไม่รองรับ")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_fps = max(1.0, fps / stride)

    out_path = os.path.join(tempfile.mkdtemp(), "result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (width, height))

    total_counts: Counter = Counter()       # รวมจำนวนตรวจจับสะสมต่อคลาส
    peak_counts: dict[str, int] = defaultdict(int)  # จำนวนสูงสุดที่เจอพร้อมกันในเฟรมเดียว
    processed = 0
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                res = model.predict(frame, conf=float(conf), device=device, verbose=False)[0]
                frame_classes = Counter()
                for b in res.boxes:
                    name = model.names[int(b.cls)]
                    name = PROMPT_TO_TH.get(name, name)  # แปลเป็นป้ายไทยถ้ามี
                    frame_classes[name] += 1
                total_counts.update(frame_classes)
                for name, c in frame_classes.items():
                    peak_counts[name] = max(peak_counts[name], c)

                writer.write(res.plot())  # เฟรมที่วาดกรอบแล้ว (BGR)
                processed += 1

                if total:
                    progress(min(idx / total, 0.99),
                             desc=f"กำลังประมวลผล เฟรม {idx}/{total}")
            idx += 1
    finally:
        cap.release()
        writer.release()

    progress(0.99, desc="กำลังเข้ารหัสวิดีโอผลลัพธ์...")
    final_path = _reencode_h264(out_path)

    # ตารางสรุป
    rows = [
        {"วัตถุ (class)": name,
         "ตรวจจับรวม (ทุกเฟรม)": total_counts[name],
         "สูงสุดต่อเฟรม": peak_counts[name]}
        for name in sorted(total_counts, key=lambda n: -total_counts[n])
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["วัตถุ (class)", "ตรวจจับรวม (ทุกเฟรม)", "สูงสุดต่อเฟรม"])

    summary = (
        f"✅ เสร็จสิ้น | โมเดล: `{mode_label}` | "
        f"ประมวลผล {processed} เฟรม (จาก {total or '?'}) | "
        f"อุปกรณ์: {'GPU — ' + GPU_NAME if device == 0 else 'CPU'} | "
        f"พบวัตถุ {len(total_counts)} ชนิด"
    )
    return final_path, df, summary


THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size="lg",
)

_dot = "#34d399" if HAS_CUDA else "#94a3b8"
_badge = ("⚡ GPU พร้อม — " + GPU_NAME) if HAS_CUDA else "🖥️ โหมด CPU"

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;}
footer {display:none !important;}

.app-hero {
  display:flex; align-items:center; gap:18px;
  padding:26px 30px; border-radius:20px; margin-bottom:6px;
  background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 48%,#d946ef 100%);
  color:#fff; box-shadow:0 14px 38px rgba(124,58,237,.35);
}
.app-hero .hero-icon {font-size:44px; line-height:1; filter:drop-shadow(0 3px 6px rgba(0,0,0,.25));}
.app-hero h1 {margin:0; font-size:27px; font-weight:800; color:#fff; letter-spacing:.2px;}
.app-hero p  {margin:5px 0 0; opacity:.92; font-size:14px;}
.app-hero .gpu-badge {
  margin-left:auto; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
  padding:9px 16px; border-radius:999px; font-size:13px; font-weight:600;
  display:flex; align-items:center; gap:9px; white-space:nowrap; backdrop-filter:blur(8px);
}
.gpu-badge .dot {width:9px; height:9px; border-radius:50%; background:%DOT%;
  box-shadow:0 0 0 4px rgba(52,211,153,.25);}

.card {border-radius:18px !important; box-shadow:0 6px 22px rgba(15,23,42,.07) !important;
  border:1px solid var(--border-color-primary) !important; padding:14px 16px !important;}
.section-title {font-weight:700 !important; font-size:14px !important; color:#7c3aed !important;
  margin:0 2px 2px !important;}

.run-btn {
  background:linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7) !important; border:none !important;
  color:#fff !important; font-weight:700 !important; font-size:16px !important;
  border-radius:14px !important; padding:15px !important; margin-top:4px !important;
  box-shadow:0 10px 24px rgba(124,58,237,.38) !important; transition:transform .12s ease, filter .12s ease;
}
.run-btn:hover {filter:brightness(1.07); transform:translateY(-1px);}

.status-box {border-radius:14px !important; padding:14px 16px !important;
  background:linear-gradient(135deg,#f5f3ff,#faf5ff) !important;
  border:1px solid #e9d5ff !important; font-size:14px !important;}

.hint {font-size:12.5px !important; color:#64748b !important; margin:6px 2px 0 !important;}
.quick-btn {border-radius:999px !important; font-size:12.5px !important; font-weight:600 !important;
  background:#f1f5f9 !important; border:1px solid #e2e8f0 !important; color:#475569 !important;
  min-width:0 !important; padding:7px 12px !important;}
.quick-btn:hover {background:#ede9fe !important; border-color:#c4b5fd !important; color:#6d28d9 !important;}
.preset-group label {border-radius:999px !important;}
"""
CSS = CSS.replace("%DOT%", _dot)

HERO = f"""
<div class="app-hero">
  <div class="hero-icon">🎬</div>
  <div>
    <h1>AIDC Tech Video Processor</h1>
    <p>อัปโหลดวิดีโอ → ตรวจจับวัตถุด้วย AI → ดูผลลัพธ์พร้อมสรุปการตรวจจับ</p>
  </div>
  <div class="gpu-badge"><span class="dot"></span> {_badge}</div>
</div>
"""

WELCOME = (
    '<div class="status-box">👋 ยินดีต้อนรับ — อัปโหลดวิดีโอทางซ้าย '
    'แล้วกด <b>เริ่มประมวลผล</b> ผลลัพธ์จะแสดงที่นี่</div>'
)

with gr.Blocks(title="AIDC Tech Video Processor", fill_width=False) as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        # ───────── ฝั่งซ้าย: ตั้งค่า ─────────
        with gr.Column(scale=4):
            with gr.Group(elem_classes="card"):
                gr.Markdown("📤 ไฟล์วิดีโอ", elem_classes="section-title")
                video_in = gr.Video(label="", height=210)

            with gr.Group(elem_classes="card"):
                gr.Markdown("🎯 เลือกสิ่งที่จะตรวจจับ/นับ", elem_classes="section-title")
                gr.Markdown("**ชุดสำเร็จรูป** (กดเลือกทีเดียวหลายอย่าง):",
                            elem_classes="hint")
                with gr.Row():
                    quick_btns = [gr.Button(name, size="sm", elem_classes="quick-btn")
                                  for name in QUICK_SETS]
                preset_cg = gr.CheckboxGroup(
                    choices=list(PRESETS.keys()), value=[],
                    label="หรือเลือกเองทีละอย่าง", elem_classes="preset-group")
                custom_tb = gr.Textbox(
                    label="เพิ่มเอง (พิมพ์ภาษาอังกฤษ คั่นด้วย ,)",
                    placeholder="เช่น: bridge, waterfall, helicopter",
                    info="ติ๊กหรือพิมพ์อย่างใดอย่างหนึ่ง = ใช้ AI ตรวจจับตามนั้น | "
                         "ไม่เลือกอะไรเลย = ใช้โมเดลปกติ 80 ชนิด",
                )
                model_dd = gr.Dropdown(
                    find_models(), value=find_models()[0],
                    label="โมเดล .pt (ใช้เมื่อไม่ได้เลือก/พิมพ์อะไร)")

            with gr.Accordion("⚙️ ตั้งค่าขั้นสูง", open=False, elem_classes="card"):
                conf_sl = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                    label="ความมั่นใจขั้นต่ำ (confidence)")
                stride_sl = gr.Slider(1, 10, value=1, step=1,
                                      label="ประมวลผลทุก ๆ N เฟรม (เพิ่ม = เร็วขึ้น)")
                gpu_ck = gr.Checkbox(value=HAS_CUDA, label="ใช้ GPU เร่งความเร็ว",
                                     interactive=HAS_CUDA)

            run_btn = gr.Button("▶️  เริ่มประมวลผล", elem_classes="run-btn")

        # ───────── ฝั่งขวา: ผลลัพธ์ ─────────
        with gr.Column(scale=6):
            status = gr.HTML(WELCOME)
            with gr.Group(elem_classes="card"):
                gr.Markdown("🎥 วิดีโอผลลัพธ์", elem_classes="section-title")
                video_out = gr.Video(label="", height=360)
            with gr.Group(elem_classes="card"):
                gr.Markdown("📊 สรุปการตรวจจับ", elem_classes="section-title")
                table_out = gr.Dataframe(label="", interactive=False, wrap=True)

    # ปุ่มชุดสำเร็จรูป → เพิ่มรายการเข้า checkbox (รวมกับที่เลือกไว้เดิม)
    def _add_set(to_add):
        def fn(current):
            return gr.update(value=list(dict.fromkeys((current or []) + to_add)))
        return fn

    for btn, name in zip(quick_btns, QUICK_SETS):
        btn.click(_add_set(QUICK_SETS[name]), inputs=preset_cg, outputs=preset_cg)

    def _run(*args, progress=gr.Progress()):
        video, table, summary = process_video(*args, progress=progress)
        html = f'<div class="status-box">{summary}</div>'
        return video, table, html

    run_btn.click(
        _run,
        inputs=[video_in, model_dd, preset_cg, custom_tb, conf_sl, gpu_ck, stride_sl],
        outputs=[video_out, table_out, status],
    )


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, theme=THEME, css=CSS)
