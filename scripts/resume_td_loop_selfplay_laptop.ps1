[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RunId,

  [switch]$DryRun,

  [ValidateRange(25.0, 100.0)]
  [double]$CpuTargetPercent = 60.0,

  [ValidateRange(0, 64)]
  [int]$ReserveLogicalCores = 2,

  [string[]]$ResumeArgs = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_training_common.ps1")

$repoRoot = Get-MagnateRepoRoot -ScriptRoot $PSScriptRoot
$cacheInfo = Initialize-MagnateLocalCaches -RepoRoot $repoRoot
$python = Get-MagnateVenvPython -RepoRoot $repoRoot
$runtime = Assert-MagnateNode22Runtime -RepoRoot $repoRoot
$cpuProfile = Get-MagnateLaptopRuntimeProfile `
  -CpuTargetPercent $CpuTargetPercent `
  -ReserveLogicalCores $ReserveLogicalCores

Write-Host "[laptop] node=$($runtime.NodeVersion)"
Write-Host "[laptop] bridge=$($runtime.TsxPath)"
Write-Host "[laptop] python=$python"
Write-Host "[laptop] runId=$RunId"
Write-Host "[laptop] tempDir=$($cacheInfo.TempDir)"
Write-Host "[laptop] threadCaps=OMP=$env:OMP_NUM_THREADS MKL=$env:MKL_NUM_THREADS OPENBLAS=$env:OPENBLAS_NUM_THREADS NUMEXPR=$env:NUMEXPR_NUM_THREADS"
Write-Host "[laptop] cpuProfile=logical=$($cpuProfile.LogicalProcessors) physical=$($cpuProfile.PhysicalCores) targetPercent=$($cpuProfile.CpuTargetPercent) reserveLogicalCores=$($cpuProfile.ReserveLogicalCores) stageSlots=$($cpuProfile.StageSlots)"
Write-Host "[laptop] runtimeProfile=collectWorkers=$($cpuProfile.CollectWorkers) evalWorkers=$($cpuProfile.EvalWorkers) incumbentEvalWorkers=$($cpuProfile.IncumbentEvalWorkers) trainThreads=$($cpuProfile.TrainThreads) trainInteropThreads=$($cpuProfile.TrainInteropThreads)"

$command = @(
  $python,
  "-m",
  "scripts.resume_td_loop_selfplay",
  "--run-id",
  $RunId,
  "--collect-workers",
  "$($cpuProfile.CollectWorkers)",
  "--collect-progress-every-games",
  "3",
  "--train-num-threads",
  "$($cpuProfile.TrainThreads)",
  "--train-num-interop-threads",
  "$($cpuProfile.TrainInteropThreads)",
  "--eval-workers",
  "$($cpuProfile.EvalWorkers)",
  "--incumbent-eval-workers",
  "$($cpuProfile.IncumbentEvalWorkers)",
  "--progress-heartbeat-minutes",
  "30"
) + $ResumeArgs

$exitCode = Invoke-MagnateLoggedCommand `
  -RepoRoot $repoRoot `
  -LogStem "resume_td_loop_selfplay_laptop" `
  -Command $command `
  -DryRun:$DryRun

exit $exitCode
