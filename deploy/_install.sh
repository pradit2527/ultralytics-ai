set -e
cd ~/yoloe
source venv/bin/activate
python -m pip install --upgrade pip -q

echo "=== [1/2] installing torch + torchvision (cu128, aarch64) ==="
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== [2/2] installing app dependencies ==="
pip install -r deploy/requirements-dgx.txt

echo "=== installed versions ==="
python - <<'PY'
import torch, torchvision, ultralytics, gradio, cv2, numpy, pandas
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("ultralytics ", ultralytics.__version__)
print("gradio      ", gradio.__version__)
print("opencv      ", cv2.__version__)
print("numpy       ", numpy.__version__)
print("pandas      ", pandas.__version__)
PY
echo "=== DONE ==="
