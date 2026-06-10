==================================================================
 วิธีติดตั้ง Ultralytics YOLO ที่ D:\yoloe  (pip + venv + NVIDIA CUDA)
==================================================================

สิ่งที่ต้องมีก่อน
------------------
1) ติดตั้ง Python 3.10 - 3.12 จาก https://www.python.org/downloads/
   *** ตอนติดตั้ง ให้ติ๊กช่อง "Add python.exe to PATH" ด้วย ***
2) มีไดรเวอร์ NVIDIA ล่าสุด (ดาวน์โหลด: https://www.nvidia.com/Download/index.aspx)
   ไม่ต้องติดตั้ง CUDA Toolkit แยก เพราะ PyTorch มี CUDA มาในตัวแล้ว

ขั้นตอนติดตั้ง (ง่ายสุด)
------------------------
1) ดับเบิลคลิกไฟล์  install_ultralytics.bat
   (หรือคลิกขวา > Run as administrator หากเจอปัญหาสิทธิ์)
2) รอจนสคริปต์ทำงานเสร็จ มันจะ:
   - สร้าง virtual environment ที่ D:\yoloe\venv
   - ติดตั้ง PyTorch เวอร์ชัน CUDA + torchvision
   - ติดตั้ง ultralytics
   - ตรวจสอบและแสดงผลว่าเห็น GPU หรือไม่

ติดตั้งด้วยตัวเอง (ถ้าอยากพิมพ์คำสั่งเอง)
-----------------------------------------
เปิด Command Prompt แล้วพิมพ์ทีละบรรทัด:

    cd /d D:\yoloe
    py -3 -m venv venv
    venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install -U ultralytics
    yolo checks

การใช้งานครั้งต่อไป
-------------------
ทุกครั้งที่จะใช้งาน ให้เปิด terminal แล้ว activate ก่อน:

    D:\yoloe\venv\Scripts\activate

ทดสอบรันโมเดล (ตัวอย่างจาก quickstart):

    yolo predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg

หรือใน Python:

    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")
    results = model("https://ultralytics.com/images/bus.jpg")

แก้ปัญหาที่พบบ่อย
-----------------
- ถ้าผลตรวจขึ้น "CUDA available: False":
    เปิด install_ultralytics.bat แก้บรรทัด  set "TORCH_CUDA=cu124"
    เปลี่ยนเป็น cu121 หรือ cu128 แล้วลบโฟลเดอร์ venv เดิม จากนั้นรันใหม่
- ถ้าขึ้นว่าไม่พบ Python:
    ติดตั้ง Python และอย่าลืมติ๊ก "Add to PATH" แล้วเปิด terminal ใหม่
- ถ้าไม่มี NVIDIA GPU:
    เปลี่ยน index-url เป็น  https://download.pytorch.org/whl/cpu

อ้างอิง: https://docs.ultralytics.com/quickstart
