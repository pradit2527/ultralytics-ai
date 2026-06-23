"""
ASGI entry point ของ web v2.

ค่าเริ่มต้น = เสิร์ฟ Django อย่างเดียว (Gradio รันเป็นบริการแยกแล้วฝังผ่าน iframe)

ทางเลือก "รวมโปรเซสเดียว" (Django + Gradio ด้วย uvicorn ตัวเดียว):
ดูตัวอย่างในไฟล์ README.md หัวข้อ "ออปชัน: รวมเป็นโปรเซสเดียว"
"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_asgi_application()
