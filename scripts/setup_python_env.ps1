[CmdletBinding()]
param(
  [switch]$SkipTorchVerification
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_training_common.ps1")

$repoRoot = Get-MagnateRepoRoot -ScriptRoot $PSScriptRoot
$cacheInfo = Initialize-MagnateLocalCaches -RepoRoot $repoRoot
$venvPath = Join-Path $repoRoot ".venv"
$requirementsPath = Join-Path $repoRoot "requirements-dev.txt"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue

if ($null -eq $pyLauncher) {
  throw "Python launcher 'py' was not found. Install Python 3.12+ with the Windows launcher enabled."
}

Write-Host "Using temp/cache dirs under the repo:"
Write-Host "  temp:      $($cacheInfo.TempDir)"
Write-Host "  pip cache: $($cacheInfo.PipCacheDir)"
Write-Host ""

if (-not (Test-Path $venvPath)) {
  Write-Host "Creating virtual environment at $venvPath"
  & $pyLauncher.Source -3.12 -m venv $venvPath
} else {
  Write-Host "Virtual environment already exists at $venvPath"
}

Write-Host "Upgrading pip in project virtual environment"
& $venvPython -m pip install --upgrade pip

Write-Host "Installing Python dependencies from $requirementsPath with CPU-only PyTorch wheels"
& $venvPython -m pip install `
  --no-cache-dir `
  --index-url https://download.pytorch.org/whl/cpu `
  --extra-index-url https://pypi.org/simple `
  -r $requirementsPath

if (-not $SkipTorchVerification) {
  Write-Host "Verifying PyTorch runtime"
  $torchReport = & $venvPython -c "import json, torch; print(json.dumps({'version': torch.__version__, 'cuda': bool(torch.cuda.is_available())}))"
  Write-Host "  torch: $torchReport"
}

Write-Host ""
Write-Host "Python environment is ready."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Python lint: .\.venv\Scripts\python.exe -m ruff check scripts trainer trainer_tests"
Write-Host "Use Node 22.12.0+ in this shell before running yarn or training wrappers: nvm use 22.12.0"
Write-Host "JS dependencies remain a separate step: yarn install"
