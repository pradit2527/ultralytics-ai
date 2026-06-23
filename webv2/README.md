# เว็บ v2 — Django + Gradio (AIDC Tech Video Analytics)

เว็บเวอร์ชัน 2 ที่นำ **Django** เข้ามาใช้ร่วมกับเครื่องมือ **Gradio** เดิม
(`app.py` ในโฟลเดอร์แม่) โดยไม่แก้โค้ดเดิม

## แนวคิดสถาปัตยกรรม

```
                    ┌─────────────────────────────────────┐
ผู้ใช้ ──► Django (เว็บหลัก :8000)                          │
          • ระบบ login / สิทธิ์ผู้ใช้ (django.contrib.auth) │
          • หน้าผู้ดูแลระบบ (/admin) จัดการบัญชี             │
          • เก็บประวัติผลวิเคราะห์ลงฐานข้อมูล                 │
          • หน้า dashboard ฝัง ↓ ผ่าน <iframe>              │
                    └──────────────┬──────────────────────┘
                                   ▼
                    Gradio (เครื่องมือประมวลผล :7860 = app.py เดิม)
```

Django ทำสิ่งที่ Gradio ทำไม่ได้ (auth, จัดการผู้ใช้, ฐานข้อมูล, ประวัติ)
ส่วน Gradio ยังทำหน้าที่ประมวลผลวิดีโอ + แสดง progress เหมือนเดิม

## โครงสร้าง

```
webv2/
├── manage.py
├── requirements.txt
├── config/            # โปรเจกต์ Django (settings, urls, asgi, wsgi)
├── core/              # แอปหลัก: views, models (AnalysisRecord), admin, urls
├── templates/         # base, home, dashboard (iframe), history, login
└── static/css/app.css # ดีไซน์ indigo/violet เข้าชุดกับ Gradio v1
```

## ติดตั้งและรัน (Windows)

ต้องรัน **2 บริการคู่กัน**:

**1) บริการ Gradio (เครื่องมือประมวลผล)** — จากโฟลเดอร์แม่ `D:\yoloe`
```powershell
.\venv\Scripts\python.exe app.py        # ขึ้นที่ http://127.0.0.1:7860
```

**2) เว็บ v2 (Django)** — จากโฟลเดอร์ `D:\yoloe\webv2`
```powershell
.\run_webv2.ps1                          # ขึ้นที่ http://127.0.0.1:8000
```
หรือทำเองทีละขั้น:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py createsuperuser   # สร้างบัญชีผู้ดูแล
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

เปิด <http://127.0.0.1:8000> → เข้าสู่ระบบ → หน้า "เครื่องมือวิเคราะห์"
จะฝังเครื่องมือ Gradio ไว้ในหน้าเดียว

## การตั้งค่า (ผ่าน environment variable)

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `GRADIO_URL` | `http://127.0.0.1:7860` | ที่อยู่บริการ Gradio ที่จะฝังใน iframe |
| `DJANGO_SECRET_KEY` | (ค่า dev) | **ต้องตั้งค่าจริงก่อนขึ้น production** |
| `DJANGO_DEBUG` | `1` | ตั้ง `0` เมื่อใช้งานจริง |
| `DJANGO_ALLOWED_HOSTS` | `*` | คั่นด้วย `,` เช่น `172.10.1.14,localhost` |

## ออปชัน: รวมเป็นโปรเซสเดียว (ขั้นสูง)

ถ้าต้องการรัน Django + Gradio ด้วย uvicorn ตัวเดียว ใช้ FastAPI เป็นเปลือก
แล้ว mount ทั้งคู่ (ต้องลง `uvicorn`, `fastapi` และให้ `app.py` import `demo` ได้):

```python
# combined_asgi.py
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from fastapi import FastAPI
from django.core.asgi import get_asgi_application
import gradio as gr
from app import demo            # ออบเจกต์ Blocks จาก app.py เดิม

app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/analyze")   # Gradio ที่ /analyze
app.mount("/", get_asgi_application())                  # Django ครอบที่เหลือ
# รัน:  uvicorn combined_asgi:app --host 0.0.0.0 --port 8000
```
แนวทาง iframe (ค่าเริ่มต้นด้านบน) ตั้งง่ายและเสถียรกว่า เหมาะกับเริ่มต้น

## หมายเหตุ

- ฐานข้อมูลเริ่มต้นเป็น SQLite (`db.sqlite3`) — เปลี่ยนเป็น PostgreSQL ได้ใน
  `config/settings.py`
- การฝัง iframe ใช้ได้เพราะ Gradio ไม่ได้ตั้ง `X-Frame-Options: DENY`
  หากเสิร์ฟ Gradio หลัง reverse proxy ให้ตั้งค่า header ให้อนุญาตการฝัง
- โค้ด `app.py` เดิมไม่ถูกแตะต้อง — เว็บ v2 ทำงานแยกอิสระ
