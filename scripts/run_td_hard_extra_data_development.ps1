[CmdletBinding()]
param(
  [switch]$DryRun,
  [ValidateRange(0.0, 1440.0)]
  [double]$HeartbeatMinutes = 10.0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_training_common.ps1")

$repoRoot = Get-MagnateRepoRoot -ScriptRoot $PSScriptRoot
$cacheInfo = Initialize-MagnateLocalCaches -RepoRoot $repoRoot
$python = Get-MagnateVenvPython -RepoRoot $repoRoot
$experimentId = "td-hard-extra-data-continuation-v1"

if (-not $DryRun) {
  $activeProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    (
      $_.CommandLine -match 'scripts\.run_td_hard_extra_data_development' -or
      (
        $_.CommandLine -match 'scripts\.evaluate_td_replay_holdout' -and
        $_.CommandLine -match $experimentId
      )
    )
  }
  if ($null -ne $activeProcesses) {
    $processIds = @($activeProcesses | ForEach-Object ProcessId) -join ", "
    throw "An extra-data development runner or evaluator is already active (process IDs: $processIds). Refusing to launch a duplicate."
  }
}

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"

$command = @(
  $python,
  "-m",
  "scripts.run_td_hard_extra_data_development",
  "--repo-root",
  $repoRoot,
  "--python-bin",
  $python,
  "--heartbeat-minutes",
  "$HeartbeatMinutes"
)
if ($DryRun) {
  $command += "--dry-run"
}

Write-Host "[extra-data-dev] python=$python"
Write-Host "[extra-data-dev] tempDir=$($cacheInfo.TempDir)"
Write-Host "[extra-data-dev] threadCaps=OMP=$env:OMP_NUM_THREADS MKL=$env:MKL_NUM_THREADS OPENBLAS=$env:OPENBLAS_NUM_THREADS NUMEXPR=$env:NUMEXPR_NUM_THREADS"
Write-Host "[extra-data-dev] execution=sequential evaluatorThreads=4 finalTestAccess=false"

Push-Location $repoRoot
try {
  $exitCode = Invoke-MagnateLoggedCommand `
    -RepoRoot $repoRoot `
    -LogStem "run_td_hard_extra_data_development" `
    -Command $command
} finally {
  Pop-Location
}

exit $exitCode
