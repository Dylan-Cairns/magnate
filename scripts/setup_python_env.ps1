$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPath = Join-Path $repoRoot ".venv"
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$venvPython = Join-Path $venvPath "Scripts\\python.exe"

if (-not (Test-Path $venvPath)) {
  Write-Host "Creating virtual environment at $venvPath"
  py -3.12 -m venv $venvPath
} else {
  Write-Host "Virtual environment already exists at $venvPath"
}

Write-Host "Upgrading pip in project virtual environment"
& $venvPython -m pip install --upgrade pip

Write-Host "Installing Python dependencies from $requirementsPath"
& $venvPython -m pip install -r $requirementsPath

Write-Host ""
Write-Host "Python environment is ready."
Write-Host "Activate with: .\\.venv\\Scripts\\Activate.ps1"
