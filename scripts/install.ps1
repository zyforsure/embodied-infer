$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$Venv = if ($env:VENV) { $env:VENV } else { Join-Path $Root ".venv" }

& $PythonBin -m venv $Venv
$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e "$Root[test]"
Write-Host "Installed embodied-infer in $Venv"
Write-Host "Try: $(Join-Path $Venv 'Scripts\embodied-infer.exe') sim"
