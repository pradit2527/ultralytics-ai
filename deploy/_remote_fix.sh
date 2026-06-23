set -e
SRC="/home/newuser/C:/Program Files/Git/home/newuser/yoloe"
if [ -d "$SRC" ]; then
  cp -rf "$SRC"/. ~/yoloe/
  rm -rf "/home/newuser/C:"
  echo "=== moved ==="
fi
echo "=== ~/yoloe ==="
ls -lh ~/yoloe
echo "=== ~/yoloe/deploy ==="
ls -lh ~/yoloe/deploy
echo "=== leftover check ==="
ls -la /home/newuser/ | grep -i 'C:' || echo "clean: no C: dir"
