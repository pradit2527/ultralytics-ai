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
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd
import torch
from ultralytics import YOLO, YOLOWorld

# โมเดล open-vocabulary สำหรับตรวจจับวัตถุ "ตามคำที่พิมพ์" (เช่น tree)
WORLD_MODEL = "yolov8s-world.pt"
# โมเดล segmentation (วาด mask ขอบเขตวัตถุ) — รุ่นเดียวกับ yolo26n ดาวน์โหลดอัตโนมัติ
SEG_MODEL = "yolo26n-seg.pt"

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


def load_animals() -> list[str]:
    """โหลดรายชื่อสัตว์จากไฟล์ animals.txt (ใช้เป็นตัวเลือกตรวจจับแบบละเอียด)"""
    p = ROOT / "animals.txt"
    if p.exists():
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    return []


ANIMALS = load_animals()


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


def _ffmpeg_exe():
    """หา ffmpeg: ใช้ของระบบก่อน ไม่มีก็ใช้ตัวที่มากับ imageio-ffmpeg"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _reencode_h264(src: str) -> str:
    """แปลงเป็น H.264 ให้เบราว์เซอร์เล่นในหน้าเว็บได้ (เล่นปุ่ม play ได้ทันที)"""
    exe = _ffmpeg_exe()
    if not exe:
        return src
    dst = src.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            [exe, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", dst],
            check=True, capture_output=True,
        )
        return dst
    except Exception:
        return src


def counts_to_df(counts, tracking):
    """แปลงลิสต์ counts → DataFrame (เพิ่มคอลัมน์ ‘นับตัวจริง’ เมื่อเปิด tracking)"""
    base_cols = ["วัตถุ (class)", "ตรวจจับรวม (ทุกเฟรม)", "สูงสุดต่อเฟรม"]
    cols = base_cols + (["นับตัวจริง (ไม่ซ้ำ)"] if tracking else [])
    rows = []
    for c in (counts or []):
        row = {base_cols[0]: c["name"], base_cols[1]: c["total"], base_cols[2]: c["peak"]}
        if tracking:
            row["นับตัวจริง (ไม่ซ้ำ)"] = c.get("unique") if c.get("unique") is not None else "-"
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def process_video(video_path, model_path, preset_labels, animal_labels, custom_classes,
                  segment, track, conf, use_gpu, frame_stride, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("กรุณาอัปโหลดไฟล์วิดีโอก่อน")

    device = 0 if (use_gpu and HAS_CUDA) else "cpu"
    seg_note = ""
    # รวม: หมวดที่ติ๊ก + สัตว์ที่เลือกจากรายการละเอียด + ที่พิมพ์เอง (ตัดซ้ำ คงลำดับ)
    chosen = [PRESETS[l] for l in (preset_labels or []) if l in PRESETS]
    chosen += list(animal_labels or [])          # ชื่อสัตว์เป็นภาษาอังกฤษอยู่แล้ว
    chosen = list(dict.fromkeys(chosen + parse_classes(custom_classes)))
    if chosen:
        progress(0, desc=f"กำลังโหลดโมเดล AI สำหรับ: {', '.join(chosen[:8])}"
                         + (" ..." if len(chosen) > 8 else ""))
        model = get_world_model(chosen)   # ตรวจจับตามรายการที่เลือก/พิมพ์
        mode_label = f"AI ตรวจจับเอง ({len(chosen)} ชนิด)"
        if segment:
            # YOLO-World ให้ได้แค่กรอบ ไม่มี mask — แจ้งเตือนแล้วใช้กรอบตามปกติ
            seg_note = " | ⚠️ โหมด mask ใช้ได้เฉพาะตรวจจับ 80 ชนิดมาตรฐาน (ไม่รวมที่เลือกเอง)"
    elif segment:
        progress(0, desc="กำลังโหลดโมเดล segmentation ...")
        model = get_model(SEG_MODEL)      # โมเดล seg วาด mask ขอบเขตวัตถุ
        mode_label = f"{SEG_MODEL} (segmentation)"
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
    unique_ids: dict[str, set] = defaultdict(set)   # ID ที่ไม่ซ้ำต่อคลาส (โหมด tracking)
    processed = 0
    idx = 0
    best_n = -1                              # เฟรมที่เจอวัตถุมากสุด (ไว้ส่งให้ Claude)
    best_frame = None
    track_on = bool(track)
    track_active = False                     # เปิดใช้ tracking ได้จริงหรือไม่

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                if track_on:
                    # นับตัวจริงด้วย tracker (ByteTrack) — persist=False เฉพาะเฟรมแรกเพื่อรีเซ็ต
                    try:
                        res = model.track(frame, conf=float(conf), device=device,
                                          persist=(processed > 0), tracker="bytetrack.yaml",
                                          verbose=False)[0]
                        track_active = True
                    except Exception:
                        track_on = False      # โมเดลนี้ track ไม่ได้ → ถอยไปใช้ detect
                        res = model.predict(frame, conf=float(conf), device=device,
                                            verbose=False)[0]
                else:
                    res = model.predict(frame, conf=float(conf), device=device, verbose=False)[0]

                plotted = res.plot()          # เฟรมที่วาดกรอบ/mask (+ID) แล้ว (BGR)
                frame_classes = Counter()
                for b in res.boxes:
                    name = model.names[int(b.cls)]
                    name = PROMPT_TO_TH.get(name, name)  # แปลเป็นป้ายไทยถ้ามี
                    frame_classes[name] += 1
                    if track_active and b.id is not None:
                        unique_ids[name].add(int(b.id))
                total_counts.update(frame_classes)
                for name, c in frame_classes.items():
                    peak_counts[name] = max(peak_counts[name], c)

                if len(res.boxes) > best_n:    # จำเฟรมที่เด่นสุดไว้
                    best_n = len(res.boxes)
                    best_frame = plotted.copy()

                writer.write(plotted)
                processed += 1

                if total:
                    progress(min(idx / total, 0.99),
                             desc=f"กำลังประมวลผล เฟรม {idx}/{total}")
            idx += 1
    finally:
        cap.release()
        writer.release()

    # บันทึกเฟรมเด่นเป็น JPEG ไว้ให้ Claude วิเคราะห์ภายหลัง
    keyframe_path = None
    if best_frame is not None:
        keyframe_path = os.path.join(tempfile.mkdtemp(), "keyframe.jpg")
        cv2.imwrite(keyframe_path, best_frame)

    progress(0.99, desc="กำลังเข้ารหัสวิดีโอผลลัพธ์...")
    final_path = _reencode_h264(out_path)

    counts = [
        {"name": n, "total": total_counts[n], "peak": peak_counts[n],
         "unique": (len(unique_ids[n]) if track_active else None)}
        for n in sorted(total_counts, key=lambda n: -total_counts[n])
    ]
    df = counts_to_df(counts, track_active)

    track_note = ""
    if track_active:
        track_note = f" | นับตัวจริง {sum(len(s) for s in unique_ids.values())} ตัว"
    summary = (
        f"✅ เสร็จสิ้น | โมเดล: `{mode_label}` | "
        f"ประมวลผล {processed} เฟรม (จาก {total or '?'}) | "
        f"อุปกรณ์: {'GPU — ' + GPU_NAME if device == 0 else 'CPU'} | "
        f"พบวัตถุ {len(total_counts)} ชนิด{track_note}{seg_note}"
    )
    # ข้อมูลสำหรับรายงาน/ส่งออก (ใช้ตอนกดปุ่มรายงาน, Word, Excel)
    report_data = {
        "mode": mode_label,
        "processed": processed,
        "total_frames": total,
        "fps": round(fps, 1),
        "duration_sec": round((total / fps), 1) if total and fps else None,
        "counts": counts,
        "tracking": track_active,
        "keyframe": keyframe_path,
        "video": final_path,
    }
    return final_path, df, summary, report_data


# ── รายงานประเมินความเสี่ยงด้วย Claude AI ──────────────────────────────
RISK_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยวิเคราะห์ภาพจากกล้องสำหรับหน่วยงานราชการ หน้าที่คือประเมิน "
    "ความเสี่ยงและความผิดปกติ จากผลการตรวจจับวัตถุด้วย AI (YOLO) ร่วมกับภาพเฟรม "
    "ตัวอย่างที่แนบมา เขียนรายงานเป็นภาษาไทยที่เป็นทางการและกระชับ ประกอบด้วย: "
    "(1) ระดับความเสี่ยงโดยรวม — ระบุชัดเจนว่า ต่ำ / ปานกลาง / สูง "
    "(2) สิ่งที่ตรวจพบและการตีความ (3) ความผิดปกติหรือจุดที่ควรเฝ้าระวัง "
    "(4) ข้อเสนอแนะ. ใช้เฉพาะข้อมูลที่เห็นจริงในภาพและตัวเลขที่ให้มา "
    "ห้ามกุข้อมูลที่ไม่ปรากฏ หากข้อมูลไม่พอ ให้ระบุข้อจำกัดอย่างตรงไปตรงมา"
)


def _build_report_prompt(report_data) -> str:
    lines = [f"โหมดการตรวจจับ: {report_data['mode']}"]
    if report_data.get("duration_sec"):
        lines.append(f"ความยาววิดีโอโดยประมาณ: {report_data['duration_sec']} วินาที "
                     f"({report_data['fps']} fps)")
    lines.append(f"จำนวนเฟรมที่ประมวลผล: {report_data['processed']}")
    lines.append("\nสรุปวัตถุที่ตรวจพบ (เรียงตามจำนวนรวม):")
    if report_data["counts"]:
        for c in report_data["counts"]:
            lines.append(f"- {c['name']}: ตรวจจับรวมทุกเฟรม {c['total']} ครั้ง, "
                         f"สูงสุดต่อเฟรม {c['peak']}")
    else:
        lines.append("- ไม่พบวัตถุที่ตรวจจับได้")
    lines.append("\nภาพที่แนบคือเฟรมที่ตรวจพบวัตถุมากที่สุด (วาดกรอบ/mask แล้ว) "
                 "โปรดประเมินความเสี่ยงและความผิดปกติพร้อมข้อเสนอแนะ")
    return "\n".join(lines)


def generate_ai_report(report_data, progress=gr.Progress()):
    if not report_data:
        return "⚠️ กรุณากด **เริ่มประมวลผล** วิดีโอก่อน แล้วจึงสร้างรายงาน"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ("### ⚠️ ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY\n\n"
                "1. รับคีย์ที่ https://console.anthropic.com\n"
                "2. เปิด PowerShell พิมพ์ (ใส่คีย์ของคุณ):\n"
                "```\nsetx ANTHROPIC_API_KEY \"sk-ant-...\"\n```\n"
                "3. ปิดแล้วเปิดแอปใหม่ (`run_video_ui.bat`)")
    try:
        import anthropic
        import base64
    except Exception:
        return "⚠️ ไม่พบไลบรารี anthropic — ติดตั้งด้วย `pip install anthropic`"

    progress(0.3, desc="กำลังส่งให้ Claude วิเคราะห์ ...")
    content = []
    kf = report_data.get("keyframe")
    if kf and os.path.exists(kf):
        with open(kf, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": img_b64}})
    content.append({"type": "text", "text": _build_report_prompt(report_data)})

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=3000,
            thinking={"type": "adaptive"},
            system=RISK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError:
        return "⚠️ ANTHROPIC_API_KEY ไม่ถูกต้อง — โปรดตรวจสอบคีย์อีกครั้ง"
    except anthropic.APIError as e:
        return f"⚠️ เรียก Claude ไม่สำเร็จ: {e}"
    except Exception as e:
        return f"⚠️ เกิดข้อผิดพลาด: {e}"

    text = "".join(b.text for b in msg.content if b.type == "text")
    return "## 📝 รายงานประเมินความเสี่ยง (วิเคราะห์โดย Claude AI)\n\n" + text


# ── หน้า "รายงานผล" — รวบรวมผลการวิเคราะห์ล่าสุดเป็นเอกสารทางการ ──────
_TH_MONTHS = ["", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
              "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


def _thai_now() -> str:
    n = datetime.now()
    return f"{n.day} {_TH_MONTHS[n.month]} {n.year + 543} เวลา {n:%H:%M} น."


def build_report_page(report_data, ai_text):
    """คืนค่า (HTML หัว+KPI, ภาพเฟรมเด่น, DataFrame ตาราง, Markdown รายงาน AI)"""
    empty_df = counts_to_df([], False)
    if not report_data:
        html = ('<div class="status-box">ยังไม่มีผลการวิเคราะห์ — กรุณาไปที่แท็บ '
                '<b>“วิเคราะห์วิดีโอ”</b> อัปโหลดวิดีโอและกด “เริ่มประมวลผล” ก่อน '
                'แล้วจึงกลับมาที่หน้านี้</div>')
        return html, None, empty_df, ""

    counts = report_data["counts"]
    tracking = report_data.get("tracking")
    total_det = sum(c["total"] for c in counts)
    peak_all = max((c["peak"] for c in counts), default=0)
    unique_all = sum((c.get("unique") or 0) for c in counts) if tracking else None
    dur = report_data.get("duration_sec")

    def kpi(value, label):
        return (f'<div class="kpi"><div class="kpi-v">{value}</div>'
                f'<div class="kpi-l">{label}</div></div>')

    cards = [kpi(len(counts), "ชนิดวัตถุที่พบ"),
             kpi(f"{total_det:,}", "การตรวจจับรวม (ทุกเฟรม)")]
    if tracking:
        cards.append(kpi(unique_all, "นับตัวจริงรวม (ไม่ซ้ำ)"))
    cards.append(kpi(peak_all, "สูงสุดพร้อมกัน/เฟรม"))
    cards.append(kpi(report_data["processed"], "เฟรมที่ประมวลผล"))

    meta = (
        f'<div class="rpt-meta"><span><b>วันที่ออกรายงาน:</b> {_thai_now()}</span>'
        f'<span><b>โหมดตรวจจับ:</b> {report_data["mode"]}</span>'
        + (f'<span><b>ความยาววิดีโอ:</b> {dur} วินาที ({report_data["fps"]} fps)</span>'
           if dur else "")
        + '</div>'
    )
    html = (
        '<div class="rpt-doc">'
        '<div class="rpt-head"><div class="rpt-emblem">🛡️</div>'
        '<div><div class="rpt-org">AIDC TECH</div>'
        '<div class="rpt-title">รายงานผลการวิเคราะห์วิดีโอด้วยปัญญาประดิษฐ์</div></div></div>'
        + meta + f'<div class="kpi-row">{"".join(cards)}</div></div>'
    )

    df = counts_to_df(counts, tracking)
    keyframe = report_data.get("keyframe")
    keyframe = keyframe if (keyframe and os.path.exists(keyframe)) else None
    ai_md = ai_text or ("> ยังไม่ได้สร้างรายงานประเมินความเสี่ยงด้วย AI — "
                        "กดปุ่ม “สร้างรายงานประเมินความเสี่ยง (Claude AI)” "
                        "ในแท็บวิเคราะห์วิดีโอ")
    return html, keyframe, df, ai_md


# ── ส่งออกเอกสาร: Word / Excel / วิดีโอ ────────────────────────────────
def export_excel(report_data):
    if not report_data:
        raise gr.Error("ยังไม่มีผลการวิเคราะห์ — กรุณาประมวลผลก่อน")
    df = counts_to_df(report_data["counts"], report_data.get("tracking"))
    path = os.path.join(tempfile.mkdtemp(), "AIDC_detection_report.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="สรุปการตรวจจับ")
    return path


def export_word(report_data, ai_text):
    if not report_data:
        raise gr.Error("ยังไม่มีผลการวิเคราะห์ — กรุณาประมวลผลก่อน")
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Tahoma"          # ฟอนต์รองรับภาษาไทย
    style.font.size = Pt(14)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:cs"), "Tahoma")

    org = doc.add_paragraph()
    org.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = org.add_run("AIDC TECH"); r.bold = True; r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0xC9, 0xA1, 0x4A)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("รายงานผลการวิเคราะห์วิดีโอด้วยปัญญาประดิษฐ์")
    r.bold = True; r.font.size = Pt(18); r.font.color.rgb = RGBColor(0x0E, 0x2A, 0x5E)

    for line in [f"วันที่ออกรายงาน: {_thai_now()}",
                 f"โหมดตรวจจับ: {report_data['mode']}",
                 (f"ความยาววิดีโอ: {report_data['duration_sec']} วินาที "
                  f"({report_data['fps']} fps)" if report_data.get('duration_sec') else None),
                 f"จำนวนเฟรมที่ประมวลผล: {report_data['processed']}"]:
        if line:
            doc.add_paragraph(line)

    doc.add_paragraph()
    h = doc.add_paragraph(); r = h.add_run("สรุปการตรวจจับ"); r.bold = True; r.font.size = Pt(15)

    counts = report_data["counts"]
    tracking = report_data.get("tracking")
    headers = ["วัตถุ", "ตรวจจับรวม (ทุกเฟรม)", "สูงสุดต่อเฟรม"] + \
              (["นับตัวจริง (ไม่ซ้ำ)"] if tracking else [])
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    for i, htext in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = htext
        for p in cell.paragraphs:
            for rr in p.runs:
                rr.bold = True
    for c in counts:
        cells = table.add_row().cells
        cells[0].text = str(c["name"])
        cells[1].text = str(c["total"])
        cells[2].text = str(c["peak"])
        if tracking:
            cells[3].text = str(c.get("unique") if c.get("unique") is not None else "-")

    kf = report_data.get("keyframe")
    if kf and os.path.exists(kf):
        doc.add_paragraph()
        h = doc.add_paragraph(); r = h.add_run("ภาพเฟรมตัวอย่าง (ตรวจพบวัตถุมากที่สุด)")
        r.bold = True; r.font.size = Pt(15)
        try:
            doc.add_picture(kf, width=Inches(6))
        except Exception:
            pass

    if ai_text:
        doc.add_paragraph()
        h = doc.add_paragraph(); r = h.add_run("รายงานประเมินความเสี่ยง (วิเคราะห์โดย AI)")
        r.bold = True; r.font.size = Pt(15)
        for ln in ai_text.splitlines():
            ln = ln.lstrip("#").replace("**", "").strip()
            if ln:
                doc.add_paragraph(ln)

    doc.add_paragraph()
    foot = doc.add_paragraph("สงวนลิขสิทธิ์ © 2569 AIDC Tech · เพื่อการใช้งานภายในหน่วยงาน")
    foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for rr in foot.runs:
        rr.font.size = Pt(10); rr.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    path = os.path.join(tempfile.mkdtemp(), "AIDC_report.docx")
    doc.save(path)
    return path


def get_result_video(report_data):
    v = (report_data or {}).get("video")
    if not v or not os.path.exists(v):
        raise gr.Error("ยังไม่มีวิดีโอผลลัพธ์ — กรุณาประมวลผลก่อน")
    return v


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
:root {
  --navy:#0e2a5e; --navy-700:#143a73; --navy-600:#16407f; --navy-900:#0b2350;
  --gold:#c9a14a; --gold-soft:#e3c97e; --ink:#1f2a44; --muted:#5b6b86;
  --line:#dde4ef; --page:#eef1f6;
}
.gradio-container {max-width:1240px !important; margin:auto !important;
  background:var(--page) !important; padding:18px 16px 0 !important;}
footer {display:none !important;}
.gradio-container * {font-feature-settings:"liga" 1;}

/* ── แถบ utility ด้านบน ───────────────────────────────── */
.app-ubar {display:flex; align-items:center; justify-content:space-between;
  background:var(--navy-900); color:#c5d4ef; font-size:11.5px; letter-spacing:.3px;
  padding:6px 16px; border-radius:8px 8px 0 0; border-bottom:1px solid rgba(255,255,255,.08);}
.app-ubar .ub-right {display:inline-flex; align-items:center; gap:7px; color:#e8eefc; font-weight:600;}
.app-ubar .ub-dot {width:8px; height:8px; border-radius:50%; background:%DOT%;
  box-shadow:0 0 0 3px rgba(34,197,94,.22);}

/* ── masthead ─────────────────────────────────────────── */
.app-masthead {display:flex; align-items:center; gap:18px; color:#fff;
  padding:18px 26px; background:linear-gradient(180deg,var(--navy-600),var(--navy));
  border-left:1px solid rgba(255,255,255,.06); border-right:1px solid rgba(255,255,255,.06);}
.app-masthead .mh-emblem {width:60px; height:60px; flex:0 0 60px; border-radius:50%;
  background:rgba(255,255,255,.10); border:2px solid var(--gold); display:flex;
  align-items:center; justify-content:center; font-size:31px;
  box-shadow:0 0 0 5px rgba(201,161,74,.12);}
.app-masthead .org {font-size:12px; letter-spacing:3px; color:var(--gold-soft); font-weight:700;}
.app-masthead h1 {margin:3px 0 0; font-size:22px; font-weight:700; color:#fff; line-height:1.3;}
.app-masthead .sub {margin:3px 0 0; font-size:12.5px; color:#a9bce0;}
.app-masthead .mh-ver {margin-left:auto; text-align:right; font-size:12px; color:#cdd9f2;
  border-left:1px solid rgba(255,255,255,.15); padding-left:18px; line-height:1.7;}
.app-masthead .mh-ver b {color:#fff; font-size:13px;}

/* ── แถบเมนูระบบ ──────────────────────────────────────── */
.app-nav {display:flex; gap:4px; background:var(--navy-700); padding:0 14px;
  border-radius:0 0 8px 8px; border-top:3px solid var(--gold);
  box-shadow:0 8px 22px rgba(14,42,94,.22);}
.app-nav .nav-item {color:#bcd0f0; font-size:13px; font-weight:600; padding:11px 16px;
  border-bottom:3px solid transparent; margin-bottom:-3px;}
.app-nav .nav-item.active {color:#fff; border-bottom-color:var(--gold); background:rgba(255,255,255,.06);}

/* ── แถบชื่อหน้า / breadcrumb ─────────────────────────── */
.page-head {display:flex; align-items:center; justify-content:space-between;
  margin:14px 0 14px; flex-wrap:wrap; gap:8px;
  background:linear-gradient(180deg,var(--navy-600),var(--navy));
  padding:14px 20px; border-radius:9px; border-left:5px solid var(--gold);
  box-shadow:0 4px 14px rgba(14,42,94,.20);}
.page-head .pg-title {font-size:18px; font-weight:700; color:#ffffff;}
.page-head .crumb {font-size:12px; color:#dbe5f7;}
.page-head .crumb b {color:#ffffff;}

/* ── การ์ด ───────────────────────────────────────────── */
.card {border-radius:10px !important; background:#fff !important;
  border:1px solid var(--line) !important;
  box-shadow:0 1px 2px rgba(16,42,94,.04), 0 8px 24px rgba(16,42,94,.05) !important;
  padding:16px 18px !important; overflow:visible !important;}
.section-title {font-weight:700 !important; font-size:14.5px !important; color:var(--navy) !important;
  border-left:4px solid var(--gold); padding-left:10px !important;
  margin:0 0 11px !important; letter-spacing:.2px;}

/* ── ปุ่มหลัก ─────────────────────────────────────────── */
.run-btn {background:linear-gradient(180deg,var(--navy-600),var(--navy)) !important;
  border:1px solid var(--navy-900) !important; color:#fff !important; font-weight:700 !important;
  font-size:16px !important; letter-spacing:.4px; border-radius:9px !important;
  padding:15px !important; margin-top:6px !important;
  box-shadow:0 6px 16px rgba(14,42,94,.32) !important; transition:transform .12s, filter .12s;}
.run-btn:hover {filter:brightness(1.12); transform:translateY(-1px);}

.ai-btn {background:linear-gradient(180deg,#fffaf0,#fbf3df) !important;
  border:1px solid var(--gold) !important; color:#7a5f24 !important; font-weight:700 !important;
  border-radius:9px !important; padding:12px !important;}
.ai-btn:hover {background:#f7ecd3 !important;}

.status-box {border-radius:9px !important; padding:14px 16px !important; color:var(--ink) !important;
  background:linear-gradient(180deg,#f3f7fd,#eef3fb) !important; border:1px solid #cfdcf2 !important;
  border-left:4px solid var(--navy) !important; font-size:14px !important; line-height:1.7;}

.hint {font-size:12.5px !important; color:var(--muted) !important; margin:4px 2px 2px !important;}
.quick-btn {border-radius:7px !important; font-size:12.5px !important; font-weight:600 !important;
  background:#eef2f8 !important; border:1px solid #d4ddec !important; color:#274472 !important;
  min-width:0 !important; padding:7px 12px !important;}
.quick-btn:hover {background:var(--navy) !important; border-color:var(--navy) !important; color:#fff !important;}

/* ── หัวข้อหมวด + ตาราง grid ของรายการวัตถุ ───────────── */
.cat-title {font-size:12.5px !important; font-weight:700 !important; color:#274472 !important;
  margin:12px 2px 4px !important;}
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
  border-color:var(--navy) !important; background:#e8eff8 !important;
  box-shadow:inset 0 0 0 1px var(--navy) !important;}

/* ── dropdown รายการสัตว์: กันการ์ดบังป๊อปอัป ───────────── */
.animal-dd, .animal-dd * {overflow:visible !important;}
.animal-dd .options, .options {z-index:80 !important;}

/* ── แท็บระบบแบบราชการ (ใช้แทนแถบเมนู) ─────────────────── */
.gov-tabs {border-top:3px solid var(--gold) !important; border-radius:0 0 8px 8px !important;
  background:var(--navy-700) !important; box-shadow:0 8px 22px rgba(14,42,94,.22) !important;
  margin-bottom:6px !important;}
.gov-tabs .tab-nav, .gov-tabs [role="tablist"] {background:transparent !important;
  border:none !important; padding:0 10px !important; gap:2px !important;}
.gov-tabs button[role="tab"] {color:#bcd0f0 !important; font-weight:600 !important;
  font-size:13.5px !important; background:transparent !important; border:none !important;
  border-bottom:3px solid transparent !important; padding:12px 18px !important; margin-bottom:-3px !important;}
.gov-tabs button[role="tab"]:hover {color:#fff !important; background:rgba(255,255,255,.05) !important;}
.gov-tabs button[role="tab"].selected {color:#fff !important;
  border-bottom-color:var(--gold) !important; background:rgba(255,255,255,.07) !important;}

/* ── การ์ด KPI + เอกสารรายงาน ─────────────────────────── */
.rpt-doc {background:#fff; border:1px solid var(--line); border-top:4px solid var(--gold);
  border-radius:10px; padding:22px 24px; box-shadow:0 8px 24px rgba(16,42,94,.06);}
.rpt-head {display:flex; align-items:center; gap:16px; border-bottom:1px solid var(--line);
  padding-bottom:14px; margin-bottom:14px;}
.rpt-emblem {width:52px; height:52px; flex:0 0 52px; border-radius:50%; font-size:27px;
  display:flex; align-items:center; justify-content:center;
  background:#f3f6fc; border:2px solid var(--gold);}
.rpt-org {font-size:12px; letter-spacing:3px; color:var(--gold); font-weight:700;}
.rpt-title {font-size:19px; font-weight:700; color:var(--navy); margin-top:2px;}
.rpt-meta {display:flex; flex-wrap:wrap; gap:8px 26px; font-size:12.5px; color:var(--muted);
  margin-bottom:16px;}
.rpt-meta b {color:var(--navy-600); font-weight:600;}
.kpi-row {display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px;}
.kpi {background:linear-gradient(180deg,#f7f9fd,#eef3fb); border:1px solid #d8e2f0;
  border-radius:9px; padding:15px 14px; text-align:center;}
.kpi-v {font-size:26px; font-weight:800; color:var(--navy); line-height:1.1;}
.kpi-l {font-size:11.5px; color:var(--muted); margin-top:5px;}

/* ── ส่วนท้าย (footer หลายคอลัมน์) ─────────────────────── */
.app-footer {margin-top:18px; border-radius:10px; overflow:hidden;
  border:1px solid var(--line); background:#fff;}
.app-footer .ft-cols {display:flex; flex-wrap:wrap; gap:24px; padding:20px 26px;
  background:linear-gradient(180deg,#f7f9fc,#fff);}
.app-footer .ft-col {flex:1 1 200px; font-size:12.5px; color:var(--muted); line-height:1.8;}
.app-footer .ft-h {font-size:13px; font-weight:700; color:var(--navy);
  border-bottom:2px solid var(--gold); padding-bottom:5px; margin-bottom:8px; display:inline-block;}
.app-footer .ft-base {background:var(--navy-900); color:#aebfdd; font-size:11.5px;
  text-align:center; padding:10px 16px; letter-spacing:.3px;}
"""
CSS = CSS.replace("%DOT%", _dot)

HERO = f"""
<div class="app-ubar">
  <div class="ub-left">ระบบสารสนเทศองค์กร · AIDC Tech</div>
  <div class="ub-right"><span class="ub-dot"></span> {_badge}</div>
</div>
<div class="app-masthead">
  <div class="mh-emblem">🛡️</div>
  <div class="mh-titles">
    <div class="org">AIDC TECH</div>
    <h1>ລະບົບປະມວນຜົນ ແລະ ວິເຄາะວິດີໂອດ້ວຍປັນຍາປະດິດ</h1>
    <div class="sub">AIDC Tech Video Processor — AI Video Analytics System</div>
  </div>
  <div class="mh-ver"><b>เวอร์ชัน 1.0</b><br>ระบบวิเคราะห์อัจฉริยะ</div>
</div>
"""

PAGE_HEAD = """
<div class="page-head">
  <div class="pg-title">การวิเคราะห์วิดีโอด้วยปัญญาประดิษฐ์</div>
  <div class="crumb">หน้าหลัก › ระบบวิเคราะห์วิดีโอ › <b>ประมวลผล</b></div>
</div>
"""

FOOTER = """
<div class="app-footer">
  <div class="ft-cols">
    <div class="ft-col">
      <div class="ft-h">AIDC Tech</div>
      ระบบวิเคราะห์วิดีโออัจฉริยะ<br>สำหรับหน่วยงานภาครัฐ
    </div>
    <div class="ft-col">
      <div class="ft-h">เกี่ยวกับระบบ</div>
      เวอร์ชัน 1.0<br>ขับเคลื่อนด้วย Ultralytics YOLO และ Claude AI
    </div>
    <div class="ft-col">
      <div class="ft-h">การสนับสนุน</div>
      ติดต่อผู้ดูแลระบบ<br>คู่มือและเอกสารการใช้งาน
    </div>
  </div>
  <div class="ft-base">สงวนลิขสิทธิ์ &copy; 2569 AIDC Tech · เพื่อการใช้งานภายในหน่วยงาน</div>
</div>
"""

WELCOME = (
    '<div class="status-box">โปรดอัปโหลดไฟล์วิดีโอและเลือกรายการวัตถุที่ต้องการ'
    'ตรวจจับทางด้านซ้าย จากนั้นกดปุ่ม <b>เริ่มประมวลผล</b> '
    'ระบบจะแสดงผลการวิเคราะห์ ณ บริเวณนี้</div>'
)

with gr.Blocks(title="AIDC Tech Video Processor", fill_width=False) as demo:
    gr.HTML(HERO)

    report_state = gr.State(None)      # ผลตรวจจับล่าสุด (แชร์ข้ามแท็บ)
    ai_report_state = gr.State("")     # ข้อความรายงาน AI ล่าสุด

    with gr.Tabs(elem_classes="gov-tabs"):
        # ═════════ แท็บที่ 1: วิเคราะห์วิดีโอ ═════════
        with gr.Tab("วิเคราะห์วิดีโอ"):
            gr.HTML(PAGE_HEAD)
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

                        gr.Markdown(f"สัตว์ (รายการละเอียด {len(ANIMALS)} ชนิด)",
                                    elem_classes="cat-title")
                        animal_dd = gr.Dropdown(
                            choices=ANIMALS, value=[], multiselect=True, filterable=True,
                            show_label=False, elem_classes="animal-dd",
                            info="พิมพ์ค้นหาชื่อสัตว์ (ภาษาอังกฤษ) แล้วเลือกได้หลายชนิด")

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

                    seg_ck = gr.Checkbox(
                        value=False,
                        label="🎭 แสดงขอบเขตวัตถุแบบ mask (segmentation)",
                        info="วาดพื้นที่วัตถุระบายสี ไม่ใช่แค่กรอบ — ใช้ได้กับการตรวจจับ "
                             "80 ชนิดมาตรฐาน (ครั้งแรกจะดาวน์โหลดโมเดล seg เล็ก ๆ)")
                    track_ck = gr.Checkbox(
                        value=False,
                        label="🔢 นับจำนวนตัวจริง (object tracking)",
                        info="ติดตามแต่ละวัตถุด้วย ID แล้วนับ ‘ตัวที่ไม่ซ้ำ’ จริง ๆ "
                             "(แนะนำให้ตั้ง ‘ทุก ๆ N เฟรม’ = 1 เพื่อความแม่นยำ)")

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
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("รายงานวิเคราะห์ด้วย AI", elem_classes="section-title")
                        report_btn = gr.Button("📝 สร้างรายงานประเมินความเสี่ยง (Claude AI)",
                                               elem_classes="ai-btn")
                        report_md = gr.Markdown("")

        # ═════════ แท็บที่ 2: รายงานผล ═════════
        with gr.Tab("รายงานผล") as report_tab:
            gr.HTML('<div class="page-head"><div class="pg-title">รายงานผลการวิเคราะห์</div>'
                    '<div class="crumb">หน้าหลัก › <b>รายงานผล</b></div></div>')
            with gr.Row():
                load_report_btn = gr.Button("🔄 โหลดผลการวิเคราะห์ล่าสุด",
                                            elem_classes="ai-btn")
            report_page_html = gr.HTML(
                '<div class="status-box">กดปุ่ม “โหลดผลการวิเคราะห์ล่าสุด” '
                'เพื่อแสดงรายงาน (หรือสลับมาที่แท็บนี้หลังประมวลผลเสร็จ)</div>')
            with gr.Group(elem_classes="card"):
                gr.Markdown("ภาพเฟรมตัวอย่าง (ตรวจพบวัตถุมากที่สุด)",
                            elem_classes="section-title")
                report_keyframe_img = gr.Image(label="", height=320, show_label=False)
            with gr.Group(elem_classes="card"):
                gr.Markdown("ตารางสรุปการตรวจจับ", elem_classes="section-title")
                report_table = gr.Dataframe(label="", interactive=False, wrap=True)
            with gr.Group(elem_classes="card"):
                gr.Markdown("รายงานประเมินความเสี่ยง (วิเคราะห์โดย AI)",
                            elem_classes="section-title")
                report_ai_md = gr.Markdown("")
            with gr.Group(elem_classes="card"):
                gr.Markdown("ส่งออกเอกสารและไฟล์", elem_classes="section-title")
                with gr.Row():
                    word_btn = gr.Button("📄 สร้างรายงาน Word", elem_classes="ai-btn")
                    excel_btn = gr.Button("📊 สร้างตาราง Excel", elem_classes="ai-btn")
                    video_btn = gr.Button("🎬 เตรียมวิดีโอผล", elem_classes="ai-btn")
                word_file = gr.File(label="รายงาน (Word)", interactive=False)
                excel_file = gr.File(label="ตาราง (Excel)", interactive=False)
                video_file = gr.File(label="วิดีโอผลการประมวลผล", interactive=False)

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
        (animal_labels, custom_classes, segment, track,
         conf, use_gpu, stride) = rest[n_groups:]
        preset_labels = [x for vals in group_vals for x in (vals or [])]
        video_out_, table, summary, report_data = process_video(
            video, model_path, preset_labels, animal_labels, custom_classes,
            segment, track, conf, use_gpu, stride, progress=progress)
        # ล้างรายงาน AI เก่า + เก็บ report_data ไว้ใน State
        return (video_out_, table, f'<div class="status-box">{summary}</div>',
                report_data, "", "")

    run_btn.click(
        _run,
        inputs=[video_in, model_dd, *preset_groups, animal_dd, custom_tb,
                seg_ck, track_ck, conf_sl, gpu_ck, stride_sl],
        outputs=[video_out, table_out, status, report_state, report_md, ai_report_state],
    )

    def _gen_report(report_data, progress=gr.Progress()):
        text = generate_ai_report(report_data, progress)
        return text, text   # แสดงในแท็บวิเคราะห์ + เก็บไว้ใช้ในหน้ารายงาน

    report_btn.click(_gen_report, inputs=report_state,
                     outputs=[report_md, ai_report_state])

    # หน้า "รายงานผล": โหลดด้วยปุ่ม หรืออัตโนมัติเมื่อสลับมาที่แท็บ
    _report_outputs = [report_page_html, report_keyframe_img, report_table, report_ai_md]
    load_report_btn.click(build_report_page,
                          inputs=[report_state, ai_report_state], outputs=_report_outputs)
    report_tab.select(build_report_page,
                      inputs=[report_state, ai_report_state], outputs=_report_outputs)

    # ส่งออกเอกสาร/ไฟล์
    word_btn.click(export_word, inputs=[report_state, ai_report_state], outputs=word_file)
    excel_btn.click(export_excel, inputs=report_state, outputs=excel_file)
    video_btn.click(get_result_video, inputs=report_state, outputs=video_file)


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, theme=THEME, css=CSS)
