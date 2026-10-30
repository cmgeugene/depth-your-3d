import socket
import subprocess
import sys

import uvicorn

from . import config

FIREWALL_RULE = "Depth Studio {port}"


def _lan_ip() -> str | None:
    # The address the OS would use to reach other machines; connecting a UDP socket sends nothing.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def _firewall_hint(port: int):
    """On Windows, explain how to open the port if our inbound rule isn't there yet."""
    if sys.platform != "win32":
        return
    name = FIREWALL_RULE.format(port=port)
    try:
        found = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
                               capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return
    if not found:
        print("  [방화벽] 다른 PC에서 안 열리면 관리자 권한 명령 프롬프트에서 한 번만 실행하세요:")
        print(f'    netsh advfirewall firewall add rule name="{name}" dir=in action=allow protocol=TCP '
              f"localport={port} profile=private,domain")
        print('  처음 실행할 때 Windows 방화벽 창이 뜨면 "개인 네트워크"를 허용해도 됩니다.')


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # a console that can't show Korean must not stop the server
    port = config.PORT
    print(f"\n  휴먼에이 영상 처리 스튜디오  ·  이 PC: http://localhost:{port}")
    if config.HOST in ("0.0.0.0", "::"):
        ip = _lan_ip()
        print(f"  같은 네트워크의 다른 PC: http://{ip}:{port}" if ip else "  내부망 주소를 찾지 못했습니다 (ipconfig로 확인하세요).")
        _firewall_hint(port)
    elif not config.HOST.startswith("127.") and config.HOST != "localhost":
        print(f"  같은 네트워크의 다른 PC: http://{config.HOST}:{port}")
    else:
        print("  이 PC에서만 열립니다. 다른 사람과 같이 쓰려면 run-lan.bat으로 실행하세요.")
    print(flush=True)
    uvicorn.run("server.app:app", host=config.HOST, port=port, workers=1)
