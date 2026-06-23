cd ~/yoloe
source venv/bin/activate
echo "=== [1] torch CUDA on GB10 (real kernel test) ==="
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability sm_:", torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("CUDA matmul OK, checksum:", float(y.sum().abs() > 0))
PY

echo "=== [2] import cv2 (headless check) ==="
python -c "import cv2; print('cv2 OK', cv2.__version__)" 2>&1 | tail -3

echo "=== [3] YOLO inference on GPU (1 frame) ==="
python - <<'PY'
import cv2
from ultralytics import YOLO
cap = cv2.VideoCapture("87631-601466996_medium.mp4")
ok, frame = cap.read(); cap.release()
print("frame read:", ok, None if not ok else frame.shape)
m = YOLO("yolo26n.pt")
r = m.predict(frame, device=0, verbose=False)[0]
print("detections:", len(r.boxes), "| names sample:", list(m.names.values())[:5])
PY
echo "=== SMOKE DONE ==="
