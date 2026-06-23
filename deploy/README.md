# Deploy บน NVIDIA DGX Spark

แอปถูกติดตั้งและรันเป็น **systemd service** บน DGX Spark แล้ว

| รายการ | ค่า |
|---|---|
| เครื่อง | `newuser@172.10.1.14` (Ubuntu 24.04 ARM64, GPU NVIDIA GB10) |
| โฟลเดอร์โปรเจกต์ | `/home/newuser/yoloe` |
| Python env | `/home/newuser/yoloe/venv` (torch 2.11.0+cu128, CUDA บน GB10) |
| Service | `yoloe.service` (enable แล้ว — ขึ้นเองตอนเปิดเครื่อง + รีสตาร์ตเมื่อ crash) |
| **เข้าใช้งานผ่านเว็บ** | **http://172.10.1.14:7860** |

## จัดการ service จากเครื่อง Windows นี้

ใช้ตัวช่วย `deploy/dgx.py` (อ่าน credential จาก `ssh.md` อัตโนมัติ):

```powershell
$py = "D:\yoloe\venv\Scripts\python.exe"
$dgx = "D:\yoloe\deploy\dgx.py"

# ดูสถานะ
& $py $dgx run "systemctl status yoloe --no-pager | head -20"
# ดู log สด
& $py $dgx run "journalctl -u yoloe -n 50 --no-pager"
```

สั่ง start/stop/restart (ต้อง sudo — ใช้ runfile เพื่อให้แนบรหัสผ่านจาก ssh.md):
สร้างไฟล์ `.sh` สั้น ๆ เช่น `echo "$SUDO_PW" | sudo -S systemctl restart yoloe` แล้วรัน
`& $py $dgx runfile <ไฟล์.sh>`

## อัปเดตโค้ด (แก้ app.py แล้วส่งขึ้นใหม่)

```powershell
& $py $dgx put "D:\yoloe\app.py" "/home/newuser/yoloe/app.py"
# แล้ว restart service (ดูด้านบน)
```

## เปิดใช้รายงาน AI (Claude)

แก้ service ให้ใส่คีย์ แล้ว restart:
```
sudo systemctl edit --full yoloe     # ปลดคอมเมนต์บรรทัด Environment=ANTHROPIC_API_KEY=...
sudo systemctl restart yoloe
```

## เปลี่ยนพอร์ต / การเข้าถึง

แก้ `Environment=GRADIO_SERVER_PORT=` ใน unit (`/etc/systemd/system/yoloe.service`)
`GRADIO_SERVER_NAME=0.0.0.0` = เปิดให้ทุกเครื่องในเครือข่ายเข้าได้

## ไฟล์ในโฟลเดอร์นี้

- `dgx.py` — ตัวช่วย SSH/SFTP (run / runfile / put / putmany / get)
- `requirements-dgx.txt` — dependencies ฝั่ง ARM64 (ไม่รวม torch — ลงแยกจาก index cu128)
- `run_dgx.sh` — รันแอปแบบ manual (ไม่ผ่าน systemd)
- `_*.sh` — สคริปต์ที่ใช้ตอน deploy (เก็บไว้อ้างอิง)
