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

# รายการวัตถุสำเร็จรูป แบ่งเป็นหมวดหมู่: หมวด → { ป้ายไทย → คำสั่งอังกฤษ }
CATEGORIES = {
    "คนและสัตว์": {
        "👤 คน": "person",
        "🐾 สัตว์": "animal",
        "🐶 สุนัข": "dog",
        "🐦 นก": "bird",
    },
    "ธรรมชาติและภูมิประเทศ": {
        "🌳 ต้นไม้": "tree",
        "🪵 ท่อนไม้ใหญ่": "wood log",
        "🌿 พุ่มไม้": "bush",
        "🌊 ลำธาร/แม่น้ำ": "river",
        "⛰️ ภูเขา": "mountain",
        "🪨 ก้อนหิน": "rock",
    },
    "ไฟและควัน": {
        "🔥 กองไฟ/ไฟ": "fire",
        "💨 ควันไฟ": "smoke",
    },
    "ยานพาหนะและสิ่งก่อสร้าง": {
        "🚗 รถยนต์": "car",
        "🏠 อาคาร/สิ่งปลูกสร้าง": "building",
        "🛣️ ถนน/ทางเดิน": "road",
        "🚧 ป้าย/เครื่องหมาย": "sign",
    },
}
# รวมทุกหมวดเป็น dict เดียว + แผนที่ย้อนกลับ (อังกฤษ → ไทย) ไว้แปลชื่อในตาราง
PRESETS = {th: en for items in CATEGORIES.values() for th, en in items.items()}
PROMPT_TO_TH = {en: th for th, en in PRESETS.items()}

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


_WORLD_CLASSES: list[str] = []  # คลาสล่าสุดที่ตั้งไว้ (กันการตั้งซ้ำโดยไม่จำเป็น)


def get_world_model(classes: list[str]) -> YOLOWorld:
    """โมเดล open-vocabulary — ตั้งคลาสตามคำที่ผู้ใช้พิมพ์ (เช่น tree, dog)"""
    if WORLD_MODEL not in _MODEL_CACHE:
        _MODEL_CACHE[WORLD_MODEL] = YOLOWorld(WORLD_MODEL)
    model = _MODEL_CACHE[WORLD_MODEL]
    if list(classes) != _WORLD_CLASSES:
        # set_classes() ต้องเข้ารหัสข้อความด้วย CLIP ก่อน — ต้องทำบน CPU เสมอ
        # ไม่งั้นจะเจอบั๊ก "tensors on different devices": CLIP ถูกย้ายไป GPU
        # จากการ predict รอบก่อน แต่ token ข้อความถูกสร้างบน CPU แล้วชนกัน
        # วิธีแก้: ย้ายโมเดลกลับ CPU + ล้าง CLIP cache ให้สร้างใหม่บน CPU
        # (ตอน predict ระบบจะย้าย text features ไป GPU ให้เองอยู่แล้ว)
        model.to("cpu")
        try:
            model.model.clip_model = None
        except Exception:
            pass
        model.set_classes(classes)
        _WORLD_CLASSES[:] = list(classes)
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
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Sarabun"), gr.themes.GoogleFont("Inter"),
          "system-ui", "sans-serif"],
    radius_size="md",
)

_dot = "#22c55e" if HAS_CUDA else "#94a3b8"
_badge = ("พร้อมใช้งาน · GPU" if HAS_CUDA else "พร้อมใช้งาน · CPU")

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto !important;}
footer {display:none !important;}

/* ── หัวระบบแบบราชการ ─────────────────────────────────── */
.gov-header {
  display:flex; align-items:center; gap:18px;
  padding:20px 26px; margin-bottom:14px; color:#fff;
  background:linear-gradient(180deg,#16407f 0%,#0e2a5e 100%);
  border-top:4px solid #c9a14a; border-radius:10px;
  box-shadow:0 6px 18px rgba(14,42,94,.25);
}
.gov-emblem {
  width:58px; height:58px; flex:0 0 58px; border-radius:50%;
  background:rgba(255,255,255,.12); border:2px solid #c9a14a;
  display:flex; align-items:center; justify-content:center; font-size:30px;
}
.gov-titles .org {font-size:12px; letter-spacing:2.5px; text-transform:uppercase;
  color:#cfe0ff; font-weight:600;}
.gov-titles h1 {margin:2px 0 0; font-size:22px; font-weight:700; color:#fff; line-height:1.3;}
.gov-titles .sub {margin:3px 0 0; font-size:12.5px; color:#a9bce0;}
.gov-status {margin-left:auto; text-align:right; font-size:11.5px; color:#cdd9f2;
  white-space:nowrap;}
.gov-status .badge {display:inline-flex; align-items:center; gap:8px; margin-top:5px;
  background:rgba(255,255,255,.1); border:1px solid rgba(201,161,74,.55);
  padding:6px 13px; border-radius:6px; font-weight:600; color:#fff; font-size:12.5px;}
.gov-status .dot {width:8px; height:8px; border-radius:50%; background:%DOT%;
  box-shadow:0 0 0 3px rgba(34,197,94,.25);}

/* ── การ์ด ───────────────────────────────────────────── */
.card {border-radius:10px !important; background:#fff !important;
  border:1px solid #dbe2ee !important; box-shadow:0 2px 8px rgba(16,42,94,.05) !important;
  padding:14px 16px !important;}
.section-title {font-weight:700 !important; font-size:14px !important; color:#13366e !important;
  border-left:4px solid #c9a14a; padding-left:9px !important; margin:0 0 9px !important;}

/* ── ปุ่มหลัก ─────────────────────────────────────────── */
.run-btn {background:linear-gradient(180deg,#16407f,#0e2a5e) !important;
  border:1px solid #0a214b !important; color:#fff !important; font-weight:700 !important;
  font-size:15.5px !important; border-radius:8px !important; padding:14px !important;
  margin-top:4px !important; box-shadow:0 4px 12px rgba(14,42,94,.3) !important;
  transition:filter .12s ease;}
.run-btn:hover {filter:brightness(1.1);}

.status-box {border-radius:8px !important; padding:13px 16px !important; color:#1e293b !important;
  background:#eef3fb !important; border:1px solid #cfdcf2 !important;
  border-left:4px solid #13366e !important; font-size:14px !important;}

.hint {font-size:12.5px !important; color:#5b6b86 !important; margin:4px 2px 0 !important;}
.quick-btn {border-radius:6px !important; font-size:12.5px !important; font-weight:600 !important;
  background:#eef2f8 !important; border:1px solid #d4ddec !important; color:#274472 !important;
  min-width:0 !important; padding:7px 12px !important;}
.quick-btn:hover {background:#13366e !important; border-color:#13366e !important; color:#fff !important;}

/* ── หัวข้อหมวด + ตาราง grid ของรายการวัตถุ ───────────── */
.cat-title {font-size:12.5px !important; font-weight:700 !important; color:#274472 !important;
  margin:11px 2px 3px !important;}
.preset-group div[data-testid="checkbox-group"] {
  display:grid !important; grid-template-columns:repeat(2, minmax(0,1fr)) !important;
  gap:7px !important;}
.preset-group div[data-testid="checkbox-group"] label {
  margin:0 !important; width:100% !important; box-sizing:border-box !important;
  border:1px solid #d4ddec !important; border-radius:7px !important;
  background:#fbfcfe !important; padding:7px 10px !important; transition:all .12s ease;}
.preset-group div[data-testid="checkbox-group"] label:hover {
  border-color:#9db4d8 !important; background:#f1f5fb !important;}
.preset-group div[data-testid="checkbox-group"] label:has(input:checked) {
  border-color:#13366e !important; background:#e8eff8 !important;
  box-shadow:inset 0 0 0 1px #13366e !important;}

/* ── ส่วนท้าย ─────────────────────────────────────────── */
.gov-footer {text-align:center; color:#64748b; font-size:12px; line-height:1.7;
  padding:16px 0 6px; margin-top:12px; border-top:1px solid #dbe2ee;}
.gov-footer b {color:#13366e;}
"""
CSS = CSS.replace("%DOT%", _dot)

HERO = f"""
<div class="gov-header">
  <div class="gov-emblem">🛡️</div>
  <div class="gov-titles">
    <div class="org">AIDC Tech</div>
    <h1>ລະບົບປະມວນຜົນ ແລະ ວິເຄາະວິດີໂອດ້ວຍປັນຍາປະດິດ</h1>
    <div class="sub">AIDC Tech Video Processor — AI Video Analytics System</div>
  </div>
  <div class="gov-status">
    สถานะระบบประมวลผล
    <div class="badge"><span class="dot"></span> {_badge}</div>
  </div>
</div>
"""

FOOTER = """
<div class="gov-footer">
  <b>AIDC Tech</b> · ระบบวิเคราะห์วิดีโออัจฉริยะ · เวอร์ชัน 1.0<br>
  ขับเคลื่อนด้วยเทคโนโลยี Ultralytics YOLO — สงวนลิขสิทธิ์ &copy; 2569
</div>
"""

WELCOME = (
    '<div class="status-box">โปรดอัปโหลดไฟล์วิดีโอและเลือกรายการวัตถุที่ต้องการ'
    'ตรวจจับทางด้านซ้าย จากนั้นกดปุ่ม <b>เริ่มประมวลผล</b> '
    'ระบบจะแสดงผลการวิเคราะห์ ณ บริเวณนี้</div>'
)

with gr.Blocks(title="AIDC Tech Video Processor", fill_width=False) as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        # ───────── ฝั่งซ้าย: ตั้งค่า ─────────
        with gr.Column(scale=4):
            with gr.Group(elem_classes="card"):
                gr.Markdown("ไฟล์วิดีโอนำเข้า", elem_classes="section-title")
                video_in = gr.Video(label="", height=210)

            with gr.Group(elem_classes="card"):
                gr.Markdown("รายการวัตถุที่ต้องการตรวจจับ", elem_classes="section-title")
                gr.Markdown("**ชุดสำเร็จรูป** (กดเลือกทีเดียวหลายอย่าง):",
                            elem_classes="hint")
                with gr.Row():
                    quick_btns = [gr.Button(name, size="sm", elem_classes="quick-btn")
                                  for name in QUICK_SETS]
                # CheckboxGroup แยกตามหมวด เรียงเป็นตาราง grid 2 คอลัมน์
                preset_groups = []
                for cat_label, items in CATEGORIES.items():
                    gr.Markdown(cat_label, elem_classes="cat-title")
                    cg = gr.CheckboxGroup(
                        choices=list(items.keys()), value=[], show_label=False,
                        container=False, elem_classes="preset-group")
                    preset_groups.append(cg)
                custom_tb = gr.Textbox(
                    label="เพิ่มเอง (พิมพ์ภาษาอังกฤษ คั่นด้วย ,)",
                    placeholder="เช่น: bridge, waterfall, helicopter",
                    info="ติ๊กหรือพิมพ์อย่างใดอย่างหนึ่ง = ใช้ AI ตรวจจับตามนั้น | "
                         "ไม่เลือกอะไรเลย = ใช้โมเดลปกติ 80 ชนิด",
                )
                model_dd = gr.Dropdown(
                    find_models(), value=find_models()[0],
                    label="โมเดล .pt (ใช้เมื่อไม่ได้เลือก/พิมพ์อะไร)")

            with gr.Accordion("ตั้งค่าขั้นสูง", open=False, elem_classes="card"):
                conf_sl = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                    label="ความมั่นใจขั้นต่ำ (confidence)")
                stride_sl = gr.Slider(1, 10, value=1, step=1,
                                      label="ประมวลผลทุก ๆ N เฟรม (เพิ่ม = เร็วขึ้น)")
                gpu_ck = gr.Checkbox(value=HAS_CUDA, label="ใช้ GPU เร่งความเร็ว",
                                     interactive=HAS_CUDA)

            run_btn = gr.Button("เริ่มประมวลผล", elem_classes="run-btn")

        # ───────── ฝั่งขวา: ผลลัพธ์ ─────────
        with gr.Column(scale=6):
            status = gr.HTML(WELCOME)
            with gr.Group(elem_classes="card"):
                gr.Markdown("วิดีโอผลการประมวลผล", elem_classes="section-title")
                video_out = gr.Video(label="", height=360)
            with gr.Group(elem_classes="card"):
                gr.Markdown("สรุปผลการตรวจจับ", elem_classes="section-title")
                table_out = gr.Dataframe(label="", interactive=False, wrap=True)

    gr.HTML(FOOTER)

    # ปุ่มชุดสำเร็จรูป → เพิ่มรายการเข้าหมวดที่เกี่ยวข้อง (รวมกับที่เลือกไว้เดิม)
    def _add_set(to_add):
        def fn(*current_vals):
            updates = []
            for items, vals in zip(CATEGORIES.values(), current_vals):
                add_here = [x for x in to_add if x in items]
                updates.append(gr.update(
                    value=list(dict.fromkeys((vals or []) + add_here))))
            return updates
        return fn

    for btn, name in zip(quick_btns, QUICK_SETS):
        btn.click(_add_set(QUICK_SETS[name]),
                  inputs=preset_groups, outputs=preset_groups)

    n_groups = len(preset_groups)

    def _run(video, model_path, *rest, progress=gr.Progress()):
        group_vals = rest[:n_groups]
        custom_classes, conf, use_gpu, stride = rest[n_groups:]
        preset_labels = [x for vals in group_vals for x in (vals or [])]
        video_out_, table, summary = process_video(
            video, model_path, preset_labels, custom_classes, conf, use_gpu,
            stride, progress=progress)
        return video_out_, table, f'<div class="status-box">{summary}</div>'

    run_btn.click(
        _run,
        inputs=[video_in, model_dd, *preset_groups, custom_tb,
                conf_sl, gpu_ck, stride_sl],
        outputs=[video_out, table_out, status],
    )


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, theme=THEME, css=CSS)
