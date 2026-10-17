# DISK REPORT — where is the local disk going? (read-only, safe to run any time)
import subprocess, shutil

def disk_report():
    def sh(cmd):
        return subprocess.run(cmd + " 2>/dev/null", shell=True, capture_output=True, text=True).stdout.strip()
    print(sh("df -h / | tail -n 2"))
    print(f"\nfree on /: {shutil.disk_usage('/').free / 1e9:.1f} GB\n")
    for path in ("/content", "/root/.cache", "/root/.config/Google/DriveFS", "/tmp"):
        print(sh(f"du -xsh {path}") or f"{path}: not present")
    print("\nlargest folders inside the Drive cache:")
    print(sh("du -xh --max-depth=3 /root/.config/Google/DriveFS | sort -h | tail -n 8") or "none found")

disk_report()
