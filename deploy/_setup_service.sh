set -e
echo "=== sudo check ==="
echo "$SUDO_PW" | sudo -S -p '' true && echo "sudo OK"

echo "=== write systemd unit to /tmp (no sudo) ==="
cat > /tmp/yoloe.service <<'UNIT'
[Unit]
Description=AIDC Tech Video Processor (Gradio + Ultralytics YOLO)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=newuser
WorkingDirectory=/home/newuser/yoloe
Environment=GRADIO_SERVER_NAME=0.0.0.0
Environment=GRADIO_SERVER_PORT=7860
Environment=GRADIO_INBROWSER=0
# ใส่คีย์ Claude เพื่อเปิดใช้รายงาน AI (ไม่ใส่ก็รันได้ แค่ปุ่มรายงานจะแจ้งให้ตั้งคีย์):
# Environment=ANTHROPIC_API_KEY=sk-ant-...
ExecStart=/home/newuser/yoloe/venv/bin/python /home/newuser/yoloe/app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

echo "=== install + enable + start ==="
echo "$SUDO_PW" | sudo -S -p '' cp /tmp/yoloe.service /etc/systemd/system/yoloe.service
echo "$SUDO_PW" | sudo -S -p '' systemctl daemon-reload
echo "$SUDO_PW" | sudo -S -p '' systemctl enable yoloe.service
echo "$SUDO_PW" | sudo -S -p '' systemctl restart yoloe.service
echo "=== waiting for startup (model load) ==="
sleep 12
echo "$SUDO_PW" | sudo -S -p '' systemctl --no-pager status yoloe.service | head -18
echo "=== port 7860 listening? ==="
(ss -ltnp 2>/dev/null | grep 7860) || echo "NOT LISTENING YET"
echo "=== local HTTP check ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:7860/ || echo "curl failed"
echo "=== SETUP DONE ==="
