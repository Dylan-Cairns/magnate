[CmdletBinding()]
param(
  [switch]$DryRun,
  [ValidateRange(1, 32)]
  [int]$Workers = 5
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_training_common.ps1")

$repoRoot = Get-MagnateRepoRoot -ScriptRoot $PSScriptRoot
$cacheInfo = Initialize-MagnateLocalCaches -RepoRoot $repoRoot
$runtime = Assert-MagnateNode22Runtime -RepoRoot $repoRoot
$experimentId = "td-hard-extra-data-continuation-v1"
$packId = "td-hard-extra-data-primary-treatment-step-09000"
$modelIndexRelative = "model-packs-experiments/td-hard-extra-data-continuation-v1-heuristic-benchmark/index.json"
$modelIndexPath = Join-Path $repoRoot "public\model-packs-experiments\td-hard-extra-data-continuation-v1-heuristic-benchmark\index.json"
$developmentPlanPath = Join-Path $repoRoot "artifacts\training_inputs\$experimentId\development-evaluation.plan.json"
$developmentResultPath = Join-Path $repoRoot "artifacts\td_extra_data_evals\$experimentId\development\development.result.json"
$outDir = Join-Path $repoRoot "artifacts\ts-bot-evals\$experimentId-primary-treatment-step-09000-vs-heuristic-v2-medium-120"

if (-not (Test-Path -LiteralPath $developmentPlanPath)) {
  throw "Missing frozen development plan at $developmentPlanPath."
}
if (-not (Test-Path -LiteralPath $developmentResultPath)) {
  throw "Missing completed development result at $developmentResultPath."
}
if (-not (Test-Path -LiteralPath $modelIndexPath)) {
  throw "Missing isolated candidate model-pack index at $modelIndexPath."
}

$developmentResult = Get-Content -LiteralPath $developmentResultPath -Raw | ConvertFrom-Json
if (
  [string]$developmentResult.status -ne "development-completed" -or
  [bool]$developmentResult.finalTestAccessed -or
  [int]$developmentResult.selection.selectedStep -ne 9000
) {
  throw "Development result is not the frozen unsealed step-9,000 selection."
}

$developmentPlan = Get-Content -LiteralPath $developmentPlanPath -Raw | ConvertFrom-Json
$selectedRows = @(
  $developmentPlan.selectionCommands | Where-Object {
    [string]$_.role -eq "primary-treatment-selection" -and
    [int]$_.checkpointStep -eq 9000
  }
)
if ($selectedRows.Count -ne 1) {
  throw "Expected exactly one frozen primary-treatment step-9,000 command; found $($selectedRows.Count)."
}

$selectedCommand = [string[]]$selectedRows[0].command
$valueCheckpoint = Get-MagnateCommandArgument -Command $selectedCommand -Flag "--value-checkpoint"
$opponentCheckpoint = Get-MagnateCommandArgument -Command $selectedCommand -Flag "--opponent-checkpoint"
$expectedValueSha256 = Get-MagnateCommandArgument -Command $selectedCommand -Flag "--expected-value-checkpoint-sha256"
$expectedOpponentSha256 = Get-MagnateCommandArgument -Command $selectedCommand -Flag "--expected-opponent-checkpoint-sha256"
foreach ($checkpointPath in @($valueCheckpoint, $opponentCheckpoint)) {
  if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "Missing selected checkpoint at $checkpointPath."
  }
}

$actualValueSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $valueCheckpoint).Hash.ToLowerInvariant()
$actualOpponentSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $opponentCheckpoint).Hash.ToLowerInvariant()
if ($actualValueSha256 -ne $expectedValueSha256) {
  throw "Selected value checkpoint SHA-256 does not match the frozen development plan."
}
if ($actualOpponentSha256 -ne $expectedOpponentSha256) {
  throw "Selected opponent checkpoint SHA-256 does not match the frozen development plan."
}

$modelIndex = Get-Content -LiteralPath $modelIndexPath -Raw | ConvertFrom-Json
$packRows = @($modelIndex.packs | Where-Object { [string]$_.id -eq $packId })
if ([int]$modelIndex.schemaVersion -ne 1 -or $packRows.Count -ne 1) {
  throw "Isolated model-pack index does not contain exactly one $packId entry."
}
$manifestPath = Join-Path (Join-Path $repoRoot "public") ([string]$packRows[0].manifestPath -replace "/", "\")
if (-not (Test-Path -LiteralPath $manifestPath)) {
  throw "Missing selected model-pack manifest at $manifestPath."
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$weightsPath = Join-Path (Split-Path $manifestPath -Parent) ([string]$manifest.model.weightsPath)
$manifestValueCheckpoint = Resolve-MagnateProjectPath -RepoRoot $repoRoot -PathValue ([string]$manifest.source.valueCheckpoint)
$manifestOpponentCheckpoint = Resolve-MagnateProjectPath -RepoRoot $repoRoot -PathValue ([string]$manifest.source.opponentCheckpoint)
if (
  [int]$manifest.schemaVersion -ne 1 -or
  [string]$manifest.packId -ne $packId -or
  [int]$manifest.source.checkpointMetadata.step -ne 9000 -or
  $manifestValueCheckpoint -ne [System.IO.Path]::GetFullPath($valueCheckpoint) -or
  $manifestOpponentCheckpoint -ne [System.IO.Path]::GetFullPath($opponentCheckpoint) -or
  -not (Test-Path -LiteralPath $weightsPath)
) {
  throw "Candidate model pack does not match the frozen primary-treatment step-9,000 checkpoint pair."
}

$manifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
$weightsSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $weightsPath).Hash.ToLowerInvariant()
$resumeKeyMaterial = "$experimentId|primary-treatment|step=9000|value=$actualValueSha256|opponent=$actualOpponentSha256|manifest=$manifestSha256|weights=$weightsSha256"
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
  $resumeKeyBytes = [System.Text.Encoding]::UTF8.GetBytes($resumeKeyMaterial)
  $resumeKey = ([System.BitConverter]::ToString($sha256.ComputeHash($resumeKeyBytes))).Replace("-", "").ToLowerInvariant()
} finally {
  $sha256.Dispose()
}

$tsxCliPath = Join-Path $repoRoot "node_modules\tsx\dist\cli.mjs"
if (-not (Test-Path -LiteralPath $tsxCliPath)) {
  throw "Missing TSX CLI at $tsxCliPath. Run yarn install from the repo root."
}

if (-not $DryRun) {
  $activeProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    -not [string]::IsNullOrWhiteSpace([string]$_.CommandLine) -and
    [string]$_.CommandLine -match "benchmark_td_vs_heuristic_v2"
  }
  if ($null -ne $activeProcesses) {
    $processIds = @($activeProcesses | ForEach-Object ProcessId) -join ", "
    throw "A TD-vs-heuristic benchmark is already active (process IDs: $processIds). Refusing to launch a duplicate."
  }
}

$command = @(
  $runtime.NodeExecutable,
  $tsxCliPath,
  (Join-Path $repoRoot "scripts\benchmark_td_vs_heuristic_v2.ts"),
  "--games", "120",
  "--worlds", "10",
  "--rollouts", "1",
  "--depth", "40",
  "--max-root-actions", "16",
  "--rollout-epsilon", "0",
  "--td-root", "td",
  "--td-rollout", "td",
  "--td-leaf", "td",
  "--opponent", "heuristic-v2",
  "--td-model-index-path", $modelIndexRelative,
  "--td-pack-id", $packId,
  "--workers", "$Workers",
  "--max-decisions-per-game", "260",
  "--out-dir", $outDir,
  "--resume", "true",
  "--resume-key", $resumeKey
)
if ($DryRun) {
  $command += @("--dry-run", "true")
}

Write-Host "[td-vs-v2] experiment=$experimentId"
Write-Host "[td-vs-v2] candidate=$packId"
Write-Host "[td-vs-v2] checkpointStep=9000"
Write-Host "[td-vs-v2] games=120 pairedSeeds=60 workers=$Workers"
Write-Host "[td-vs-v2] opponent=heuristic-v2-medium worlds=10 rollouts=1 depth=40 maxRootActions=16 epsilon=0"
Write-Host "[td-vs-v2] node=$($runtime.NodeVersion) nodePath=$($runtime.NodeExecutable)"
Write-Host "[td-vs-v2] tempDir=$($cacheInfo.TempDir)"
Write-Host "[td-vs-v2] output=$outDir"
Write-Host "[td-vs-v2] resume=per-completed-pair"

Push-Location $repoRoot
try {
  $exitCode = Invoke-MagnateLoggedCommand `
    -RepoRoot $repoRoot `
    -LogStem "run_td_hard_extra_data_heuristic_benchmark" `
    -Command $command
} finally {
  Pop-Location
}

exit $exitCode
