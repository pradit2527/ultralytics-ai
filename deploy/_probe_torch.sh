cd ~/yoloe
source venv/bin/activate
echo "=== platform ==="
python -c "import platform; print(platform.machine(), platform.python_version())"
echo "=== cu128 torch available? ==="
pip index versions torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | head -5 || true
echo "=== cu130 torch available? ==="
pip index versions torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | head -5 || true
