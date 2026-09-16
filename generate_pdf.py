import os
import subprocess
import time

html_file = os.path.abspath("ekyc_anti_spoofing_paper.html")
pdf_file = os.path.abspath("ekyc_anti_spoofing_paper.pdf")

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_path):
    edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

if not os.path.exists(edge_path):
    raise FileNotFoundError(f"Microsoft Edge executable not found. Checked: {edge_path}")

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_file}",
    html_file
]

print(f"Compiling '{html_file}' -> '{pdf_file}'...")
result = subprocess.run(cmd, capture_output=True, text=True)

time.sleep(1.5)

if os.path.exists(pdf_file) and os.path.getsize(pdf_file) > 1000:
    print(f"SUCCESS: Generated PDF '{pdf_file}' ({os.path.getsize(pdf_file):,} bytes)")
else:
    print(f"ERROR generating PDF. Stdout: {result.stdout}, Stderr: {result.stderr}")
    exit(1)
