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

Write-Host "[laptop] node=$($runtime.NodeVersion)"
Write-Host "[laptop] bridge=$($runtime.TsxPath)"
Write-Host "[laptop] python=$python"
Write-Host "[laptop] tempDir=$($cacheInfo.TempDir)"
Write-Host "[laptop] threadCaps=OMP=$env:OMP_NUM_THREADS MKL=$env:MKL_NUM_THREADS OPENBLAS=$env:OPENBLAS_NUM_THREADS NUMEXPR=$env:NUMEXPR_NUM_THREADS"

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
  "2",
  "--train-steps",
  "20000",
  "--train-num-threads",
  "4",
  "--train-num-interop-threads",
  "1",
  "--eval-games-per-side",
  "200",
  "--eval-opponent-policy",
  "search",
  "--eval-workers",
  "2",
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
