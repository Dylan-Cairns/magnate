[CmdletBinding()]
param(
  [switch]$DryRun,
  [ValidateRange(25.0, 100.0)]
  [double]$CpuTargetPercent = 60.0,
  [ValidateRange(0, 64)]
  [int]$ReserveLogicalCores = 2,
  [string[]]$LoopArgs = @()
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
$warmStart = Get-MagnateWarmStartPair -RepoRoot $repoRoot -AllowManifestFallback

Write-Host "[laptop] node=$($runtime.NodeVersion)"
Write-Host "[laptop] bridge=$($runtime.TsxPath)"
Write-Host "[laptop] python=$python"
Write-Host "[laptop] tempDir=$($cacheInfo.TempDir)"
Write-Host "[laptop] threadCaps=OMP=$env:OMP_NUM_THREADS MKL=$env:MKL_NUM_THREADS OPENBLAS=$env:OPENBLAS_NUM_THREADS NUMEXPR=$env:NUMEXPR_NUM_THREADS"
Write-Host "[laptop] cpuProfile=logical=$($cpuProfile.LogicalProcessors) physical=$($cpuProfile.PhysicalCores) targetPercent=$($cpuProfile.CpuTargetPercent) reserveLogicalCores=$($cpuProfile.ReserveLogicalCores) stageSlots=$($cpuProfile.StageSlots)"
Write-Host "[laptop] runtimeProfile=collectWorkers=$($cpuProfile.CollectWorkers) evalWorkers=$($cpuProfile.EvalWorkers) trainThreads=$($cpuProfile.TrainThreads) trainInteropThreads=$($cpuProfile.TrainInteropThreads)"

$command = @(
  $python,
  "-m",
  "scripts.run_td_loop",
  "--run-label",
  "td-loop-r2-laptop",
  "--chunks-per-loop",
  "3",
  "--collect-games",
  "1200",
  "--collect-workers",
  "$($cpuProfile.CollectWorkers)",
  "--train-steps",
  "20000",
  "--train-num-threads",
  "$($cpuProfile.TrainThreads)",
  "--train-num-interop-threads",
  "$($cpuProfile.TrainInteropThreads)",
  "--eval-games-per-side",
  "200",
  "--eval-opponent-policy",
  "search",
  "--eval-workers",
  "$($cpuProfile.EvalWorkers)",
  "--eval-worker-torch-threads",
  "1",
  "--eval-worker-torch-interop-threads",
  "1",
  "--eval-worker-blas-threads",
  "1",
  "--progress-heartbeat-minutes",
  "30",
  "--eval-progress-log-minutes",
  "30"
)

if ($null -ne $warmStart) {
  Write-Host "[laptop] warmStartSource=$($warmStart.SourceLabel)"
  Write-Host "[laptop] warmStartRunId=$($warmStart.RunId)"
  $command += @(
    "--train-warm-start-value-checkpoint",
    $warmStart.ValuePath,
    "--train-warm-start-opponent-checkpoint",
    $warmStart.OpponentPath
  )
} else {
  Write-Host "[laptop] no warm-start checkpoint pair found; bootstrap will run from scratch."
}

$command += $LoopArgs

$exitCode = Invoke-MagnateLoggedCommand `
  -RepoRoot $repoRoot `
  -LogStem "run_td_loop_bootstrap_laptop" `
  -Command $command `
  -DryRun:$DryRun

exit $exitCode
