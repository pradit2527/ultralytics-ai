#!/usr/bin/env bash
# รันแอปบน DGX Spark — เปิดให้เข้าผ่านเครือข่ายที่พอร์ต 7860
# ใช้: bash ~/yoloe/deploy/run_dgx.sh   (หรือผ่าน systemd service ดู deploy)
set -e
cd "$(dirname "$0")/.."

source venv/bin/activate

# เปิดให้เข้าจากเครื่องอื่นในเครือข่าย + ไม่เปิดเบราว์เซอร์ (เซิร์ฟเวอร์ไม่มีจอ)
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
export GRADIO_INBROWSER=0

# ใส่คีย์ Claude ตรงนี้ถ้าต้องการให้สร้างรายงาน AI ได้ (ไม่ใส่ก็รันได้ปกติ)
# export ANTHROPIC_API_KEY=sk-ant-...

exec python app.py
