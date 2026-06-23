"""
ตัวช่วย SSH/SFTP สำหรับ deploy โปรเจกต์ไป DGX Spark
อ่าน credential จาก ssh.md (User: / PW: / IP) อัตโนมัติ

ใช้งาน:
    python dgx.py run "<shell command>"      # รันคำสั่งบน DGX
    python dgx.py put <local> <remote>        # อัปโหลดไฟล์เดียว
    python dgx.py putmany <remotedir> <f1> <f2> ...   # อัปโหลดหลายไฟล์ไปโฟลเดอร์เดียว
    python dgx.py get <remote> <local>        # ดาวน์โหลดไฟล์
"""
import sys
import os
import re
import stat
import posixpath
from pathlib import Path
from shlex import quote as shlex_quote

import paramiko

ROOT = Path(__file__).resolve().parent.parent
SSH_MD = ROOT / "ssh.md"


def creds():
    txt = SSH_MD.read_text(encoding="utf-8")
    user = re.search(r"User\s*:\s*(\S+)", txt, re.I).group(1)
    pw = re.search(r"PW\s*:\s*(\S+)", txt, re.I).group(1)
    host = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", txt).group(1)
    return host, user, pw


def connect():
    host, user, pw = creds()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, username=user, password=pw, timeout=30,
                look_for_keys=False, allow_agent=False)
    return cli


def run(cli, cmd, get_pty=False):
    """รันคำสั่ง พิมพ์ stdout/stderr แบบ stream คืนค่า exit code"""
    stdin, stdout, stderr = cli.exec_command(cmd, get_pty=get_pty, timeout=None)
    out_lines = []
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        out_lines.append(line)
    err = stderr.read().decode("utf-8", "replace")
    if err.strip():
        sys.stderr.write(err)
    rc = stdout.channel.recv_exit_status()
    return rc, "".join(out_lines), err


def exec_script(cli, script_text):
    """ส่งเนื้อสคริปต์ผ่าน stdin ไปให้ 'bash -s' — เลี่ยงปัญหา quoting ทุกชนิด"""
    chan = cli.get_transport().open_session()
    chan.settimeout(0.5)
    chan.exec_command("bash -s")
    chan.sendall(script_text.encode("utf-8"))
    chan.shutdown_write()

    def drain():
        got = True
        while got:
            got = False
            try:
                while chan.recv_ready():
                    sys.stdout.buffer.write(chan.recv(65536)); got = True
                sys.stdout.flush()
            except Exception:
                pass
            try:
                while chan.recv_stderr_ready():
                    sys.stderr.buffer.write(chan.recv_stderr(65536)); got = True
                sys.stderr.flush()
            except Exception:
                pass

    while not chan.exit_status_ready():
        drain()
        import time
        time.sleep(0.2)
    drain()  # อ่านที่ค้างหลังจบ
    return chan.recv_exit_status()


def _progress(name):
    def cb(done, total):
        if done >= total:  # พิมพ์เฉพาะตอนเสร็จ เลี่ยง output ท่วม
            print(f"  [ok] {name}: {total/1e6:.1f} MB")
    return cb


def put(cli, local, remote):
    sftp = cli.open_sftp()
    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    rdir = posixpath.dirname(remote)
    _mkdirs(sftp, rdir)
    name = posixpath.basename(remote)
    sftp.put(local, remote, callback=_progress(name))
    print()  # ขึ้นบรรทัดใหม่หลัง progress
    sftp.close()


def _mkdirs(sftp, rdir):
    if not rdir or rdir in ("/", "."):
        return
    try:
        sftp.stat(rdir)
        return  # มีอยู่แล้ว
    except IOError:
        pass
    _mkdirs(sftp, posixpath.dirname(rdir))  # สร้าง parent ก่อนถ้ายังไม่มี
    try:
        sftp.mkdir(rdir)
    except IOError:
        pass


def get(cli, remote, local):
    sftp = cli.open_sftp()
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, local, callback=_progress(posixpath.basename(remote)))
    print()
    sftp.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    action = sys.argv[1]
    cli = connect()
    try:
        if action == "run":
            rc, _, _ = run(cli, sys.argv[2])
            sys.exit(rc)
        elif action == "runfile":
            script = Path(sys.argv[2]).read_text(encoding="utf-8")
            # แนบรหัสผ่าน sudo เป็น env (จาก ssh.md) ให้สคริปต์ใช้ผ่าน $SUDO_PW
            # โดยไม่ต้องฝังรหัสในไฟล์สคริปต์ที่บันทึกไว้
            _, _, pw = creds()
            script = f"export SUDO_PW={shlex_quote(pw)}\n" + script
            sys.exit(exec_script(cli, script))
        elif action == "put":
            put(cli, sys.argv[2], sys.argv[3])
        elif action == "putmany":
            remotedir = sys.argv[2]
            for f in sys.argv[3:]:
                put(cli, f, posixpath.join(remotedir, os.path.basename(f)))
        elif action == "get":
            get(cli, sys.argv[2], sys.argv[3])
        else:
            print("unknown action:", action)
            sys.exit(1)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
