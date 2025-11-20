# run_full_system.ps1
# Launches three windows:
#  1) persistent_server (FastAPI + LSL worker)
#  2) simple_stream_demo.py (optional demo generator)
#  3) streamlit UI
#
# Edit paths below if your repository layout differs.

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projectRoot

# 1) Start persistent server
$serverCmd = "python src\inference\persistent_server.py"
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit","-Command $serverCmd" -WindowStyle Normal
Start-Sleep -Seconds 1

# 2) Start synthetic LSL demo (optional) — comment if you use a real EEG device
$demoCmd = "python simple_stream_demo.py"
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit","-Command $demoCmd" -WindowStyle Normal
Start-Sleep -Seconds 1

# 3) Start Streamlit UI
$streamlitCmd = "streamlit run src\dashboard\app.py --server.port 8502"
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit","-Command $streamlitCmd" -WindowStyle Normal

Write-Output "Launched persistent server, demo (optional), and Streamlit UI."
Write-Output "Open http://localhost:8502 for the dashboard and http://127.0.0.1:8765/status for the backend."