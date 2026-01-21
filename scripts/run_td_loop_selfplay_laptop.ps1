[CmdletBinding()]
param(
  [switch]$DryRun,
  [string[]]$LoopArgs = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_training_common.ps1")

$repoRoot = Get-MagnateRepoRoot -ScriptRoot $PSScriptRoot
$cacheInfo = Initialize-MagnateLocalCaches -RepoRoot $repoRoot
$python = Get-MagnateVenvPython -RepoRoot $repoRoot
$runtime = Assert-MagnateNode22Runtime -RepoRoot $repoRoot
$warmStart = Get-MagnateWarmStartPair -RepoRoot $repoRoot -AllowManifestFallback

if ($null -eq $warmStart) {
  throw "No warm-start checkpoint pair found. Run bootstrap first or restore models/td_checkpoints/manifest.json plus the promoted checkpoints."
}

Write-Host "[laptop] node=$($runtime.NodeVersion)"
Write-Host "[laptop] bridge=$($runtime.TsxPath)"
Write-Host "[laptop] python=$python"
Write-Host "[laptop] warmStartSource=$($warmStart.SourceLabel)"
Write-Host "[laptop] warmStartRunId=$($warmStart.RunId)"
Write-Host "[laptop] tempDir=$($cacheInfo.TempDir)"
Write-Host "[laptop] threadCaps=OMP=$env:OMP_NUM_THREADS MKL=$env:MKL_NUM_THREADS OPENBLAS=$env:OPENBLAS_NUM_THREADS NUMEXPR=$env:NUMEXPR_NUM_THREADS"

$command = @(
  $python,
  "-m",
  "scripts.run_td_loop_selfplay",
  "--run-label",
  "td-loop-selfplay-r1-laptop",
  "--chunks-per-loop",
  "18",
  "--collect-games",
  "600",
  "--collect-workers",
  "2",
  "--collect-progress-every-games",
  "10",
  "--train-steps",
  "10000",
  "--train-num-threads",
  "4",
  "--train-num-interop-threads",
  "1",
  "--train-warm-start-value-checkpoint",
  $warmStart.ValuePath,
  "--train-warm-start-opponent-checkpoint",
  $warmStart.OpponentPath,
  "--eval-games-per-side",
  "200",
  "--eval-workers",
  "2",
  "--eval-worker-torch-threads",
  "1",
  "--eval-worker-torch-interop-threads",
  "1",
  "--eval-worker-blas-threads",
  "1",
  "--incumbent-eval-games-per-side",
  "200",
  "--incumbent-eval-workers",
  "2",
  "--progress-heartbeat-minutes",
  "30",
  "--eval-progress-log-minutes",
  "30"
) + $LoopArgs

$exitCode = Invoke-MagnateLoggedCommand `
  -RepoRoot $repoRoot `
  -LogStem "run_td_loop_selfplay_laptop" `
  -Command $command `
  -DryRun:$DryRun

exit $exitCode
