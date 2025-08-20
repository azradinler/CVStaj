import os
import subprocess
import sys
import time
import webbrowser

def run() -> None:
    python_exe = sys.executable
    env = os.environ.copy()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = os.pathsep.join([base_dir, env.get("PYTHONPATH", "")])

    api_cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]

    print("Uygulama http://localhost:8000 noktasında başlatılıyor ...")
    api_proc = subprocess.Popen(api_cmd, env=env, cwd=base_dir)

    time.sleep(2.0)
    
    print("Web Arayüzü açılıyor")
    webbrowser.open("http://localhost:8000")

    try:
        while True:
            api_ret = api_proc.poll()
            if api_ret is not None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Kapanıyor...")
    finally:
        if api_proc.poll() is None:
            api_proc.terminate()
            print("Servis kapatıldı.")

if __name__ == "__main__":
    run()


