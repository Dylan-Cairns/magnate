[CmdletBinding()]
param(
  [switch]$DryRun,
  [switch]$ForcePrepare,
  [int]$Stage1Steps = 60000,
  [int]$Stage2Steps = 30000,
  [int]$SaveEverySteps = 5000,
  [int]$ProgressEverySteps = 100,
  [ValidateRange(25.0, 100.0)]
  [double]$CpuTargetPercent = 80.0,
  [ValidateRange(0, 64)]
  [int]$ReserveLogicalCores = 1,
  [double]$HeartbeatMinutes = 30.0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Missing project virtual environment at $python. Run .\scripts\setup_python_env.ps1 first."
}

$command = @(
  $python,
  "-m",
  "scripts.train_td_two_stage_imported_replay",
  "--repo-root",
  $repoRoot,
  "--python-bin",
  $python,
  "--stage1-steps",
  "$Stage1Steps",
  "--stage2-steps",
  "$Stage2Steps",
  "--save-every-steps",
  "$SaveEverySteps",
  "--progress-every-steps",
  "$ProgressEverySteps",
  "--cpu-target-percent",
  "$CpuTargetPercent",
  "--reserve-logical-cores",
  "$ReserveLogicalCores",
  "--heartbeat-minutes",
  "$HeartbeatMinutes"
)

if ($DryRun) {
  $command += "--dry-run"
}
if ($ForcePrepare) {
  $command += "--force-prepare"
}

Write-Host "[two-stage] handoff=$($command -join ' ')"
$executable = $command[0]
$arguments = @($command | Select-Object -Skip 1)
& $executable @arguments
exit $LASTEXITCODE
