"""One-click launcher — Backend + Frontend."""

import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")


def start_backend():
    """Backend'i başlat (DETACHED_PROCESS on Windows)."""
    print("[*] Backend başlatılıyor (port 8055)...")

    if sys.platform == "win32":
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8055", "--reload"],
            cwd=ROOT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8055", "--reload"],
            cwd=ROOT,
            start_new_session=True,
        )

    print(f"[+] Backend PID: {proc.pid}")
    return proc


def start_frontend():
    """Frontend dev server'ı başlat."""
    if not os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
        print("[*] Frontend bağımlılıkları yükleniyor...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, shell=True)

    print("[*] Frontend başlatılıyor (port 5180)...")

    if sys.platform == "win32":
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
            shell=True,
        )
    else:
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
        )

    print(f"[+] Frontend PID: {proc.pid}")
    return proc


if __name__ == "__main__":
    print("=" * 50)
    print("  OptiCanavar — Tam Otomatik Optimizer")
    print("=" * 50)

    backend = start_backend()
    time.sleep(2)
    frontend = start_frontend()

    print()
    print(f"  Backend:  http://localhost:8055")
    print(f"  Frontend: http://localhost:5180")
    print(f"  API Docs: http://localhost:8055/docs")
    print()
    print("  Ctrl+C ile durdurun")
    print("=" * 50)

    try:
        frontend.wait()
    except KeyboardInterrupt:
        print("\n[*] Kapatılıyor...")
        frontend.terminate()
        backend.terminate()
