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
  [int]$ReserveLogicalCores = 1
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_training_common.ps1")

function Assert-PositiveInteger {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [int]$Value
  )

  if ($Value -le 0) {
    throw "$Name must be > 0."
  }
}

function Get-ReplaySetInventory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string[]]$RunRoots,

    [Parameter(Mandatory = $true)]
    [string]$SetName
  )

  $runs = @()
  $valuePaths = @()
  $opponentPaths = @()
  $totalGames = 0
  $totalValueRows = 0
  $totalOpponentRows = 0

  foreach ($runRoot in $RunRoots) {
    $resolvedRunRoot = Resolve-MagnateProjectPath -RepoRoot $RepoRoot -PathValue $runRoot
    if (-not (Test-Path $resolvedRunRoot)) {
      throw "Missing $SetName replay run: $resolvedRunRoot"
    }

    $shardsDir = Join-Path $resolvedRunRoot "shards"
    if (-not (Test-Path $shardsDir)) {
      throw "Missing $SetName shards directory: $shardsDir"
    }

    $runValuePaths = @(Get-ChildItem -Path $shardsDir -Filter "*.value.jsonl" -File | Sort-Object Name)
    $runOpponentPaths = @(Get-ChildItem -Path $shardsDir -Filter "*.opponent.jsonl" -File | Sort-Object Name)
    $runSummaryPaths = @(Get-ChildItem -Path $shardsDir -Filter "*.summary.json" -File | Sort-Object Name)

    if ($runValuePaths.Count -eq 0) {
      throw "No value replay shards found for $SetName run: $resolvedRunRoot"
    }
    if ($runValuePaths.Count -ne $runOpponentPaths.Count) {
      throw "Value/opponent shard count mismatch for $SetName run: $resolvedRunRoot"
    }
    if ($runValuePaths.Count -ne $runSummaryPaths.Count) {
      throw "Value/summary shard count mismatch for $SetName run: $resolvedRunRoot"
    }

    $runGames = 0
    $runValueRows = 0
    $runOpponentRows = 0
    foreach ($summaryPath in $runSummaryPaths) {
      $summary = Get-Content $summaryPath.FullName -Raw | ConvertFrom-Json
      $runGames += [int]$summary.results.games
      $runValueRows += [int]$summary.results.valueTransitions
      $runOpponentRows += [int]$summary.results.opponentSamples
    }

    if ($runValueRows -ne $runOpponentRows) {
      throw "Value/opponent row count mismatch for $SetName run: $resolvedRunRoot"
    }

    $valuePaths += @($runValuePaths | ForEach-Object { $_.FullName })
    $opponentPaths += @($runOpponentPaths | ForEach-Object { $_.FullName })
    $totalGames += $runGames
    $totalValueRows += $runValueRows
    $totalOpponentRows += $runOpponentRows
    $runs += [pscustomobject]@{
      runRoot = $resolvedRunRoot
      shards = $runValuePaths.Count
      games = $runGames
      valueRows = $runValueRows
      opponentRows = $runOpponentRows
    }
  }

  if ($valuePaths.Count -ne $opponentPaths.Count) {
    throw "Value/opponent total shard count mismatch for $SetName."
  }
  if ($totalValueRows -ne $totalOpponentRows) {
    throw "Value/opponent total row count mismatch for $SetName."
  }

  return [pscustomobject]@{
    setName = $SetName
    runs = $runs
    valuePaths = [string[]]$valuePaths
    opponentPaths = [string[]]$opponentPaths
    shards = $valuePaths.Count
    games = $totalGames
    valueRows = $totalValueRows
    opponentRows = $totalOpponentRows
  }
}

function Join-JsonlFiles {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$InputPaths,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [switch]$Force
  )

  if ((Test-Path $OutputPath) -and (-not $Force)) {
    Write-Host "[two-stage] using existing replay file: $OutputPath"
    return
  }

  $outputDir = Split-Path $OutputPath -Parent
  New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
  $tempPath = "$OutputPath.tmp"
  if (Test-Path $tempPath) {
    Remove-Item -LiteralPath $tempPath -Force
  }

  Write-Host "[two-stage] writing replay file: $OutputPath"
  $outputStream = [System.IO.File]::Open(
    $tempPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
  )
  try {
    foreach ($inputPath in $InputPaths) {
      $inputStream = [System.IO.File]::Open(
        $inputPath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read
      )
      try {
        $inputStream.CopyTo($outputStream)
      } finally {
        $inputStream.Dispose()
      }
    }
  } finally {
    $outputStream.Dispose()
  }

  Move-Item -LiteralPath $tempPath -Destination $OutputPath -Force
}

function Write-TrainingInputManifest {
  param(
    [Parameter(Mandatory = $true)]
    [object]$AllInventory,

    [Parameter(Mandatory = $true)]
    [object]$HardInventory,

    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [Parameter(Mandatory = $true)]
    [string]$AllValuePath,

    [Parameter(Mandatory = $true)]
    [string]$AllOpponentPath,

    [Parameter(Mandatory = $true)]
    [string]$HardValuePath,

    [Parameter(Mandatory = $true)]
    [string]$HardOpponentPath
  )

  $payload = [pscustomobject]@{
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    curriculum = "stage1-all-data-stage2-hard-only"
    generatedReplayFiles = [pscustomobject]@{
      allValue = $AllValuePath
      allOpponent = $AllOpponentPath
      hardValue = $HardValuePath
      hardOpponent = $HardOpponentPath
    }
    allData = [pscustomobject]@{
      games = $AllInventory.games
      shards = $AllInventory.shards
      valueRows = $AllInventory.valueRows
      opponentRows = $AllInventory.opponentRows
      runs = $AllInventory.runs
    }
    hardOnly = [pscustomobject]@{
      games = $HardInventory.games
      shards = $HardInventory.shards
      valueRows = $HardInventory.valueRows
      opponentRows = $HardInventory.opponentRows
      runs = $HardInventory.runs
    }
  }

  $manifestDir = Split-Path $ManifestPath -Parent
  New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
  $payload | ConvertTo-Json -Depth 8 | Set-Content -Path $ManifestPath -Encoding UTF8
}

function Get-FinalCheckpointPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$SummaryPath,

    [Parameter(Mandatory = $true)]
    [ValidateSet("value", "opponent")]
    [string]$Kind
  )

  if (-not (Test-Path $SummaryPath)) {
    throw "Training summary not found: $SummaryPath"
  }

  $summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
  $checkpoints = @($summary.results.checkpoints)
  if ($checkpoints.Count -eq 0) {
    throw "Training summary has no checkpoints: $SummaryPath"
  }

  $last = $checkpoints[-1]
  $checkpointPath = [string]$last.$Kind
  if ([string]::IsNullOrWhiteSpace($checkpointPath)) {
    throw "Final $Kind checkpoint is missing in summary: $SummaryPath"
  }
  if (-not (Test-Path $checkpointPath)) {
    throw "Final $Kind checkpoint not found: $checkpointPath"
  }

  return (Resolve-Path $checkpointPath).Path
}

Assert-PositiveInteger -Name "Stage1Steps" -Value $Stage1Steps
Assert-PositiveInteger -Name "Stage2Steps" -Value $Stage2Steps
Assert-PositiveInteger -Name "SaveEverySteps" -Value $SaveEverySteps
Assert-PositiveInteger -Name "ProgressEverySteps" -Value $ProgressEverySteps

$repoRoot = Get-MagnateRepoRoot -ScriptRoot $PSScriptRoot
$cacheInfo = Initialize-MagnateLocalCaches -RepoRoot $repoRoot
$python = Get-MagnateVenvPython -RepoRoot $repoRoot
$cpuProfile = Get-MagnateLaptopRuntimeProfile `
  -CpuTargetPercent $CpuTargetPercent `
  -ReserveLogicalCores $ReserveLogicalCores
$warmStart = Get-MagnateWarmStartPair -RepoRoot $repoRoot -AllowManifestFallback

if ($null -eq $warmStart) {
  throw "No warm-start checkpoint pair found. Restore models/td_checkpoints/manifest.json plus promoted checkpoints first."
}

$allRunRoots = @(
  "artifacts\imported\usb-20260706\td_replay\20260619T134512Z-v2-40w180d-teacher-1400",
  "artifacts\imported\usb-20260706\artifacts\td_replay\20260619T003038Z-v2-40w180d-teacher-1000",
  "artifacts\imported\usb-20260706\artifacts\td_replay\20260618T220338Z-v2-40w180d-benchmark-4",
  "artifacts\imported\usb-20260706\artifacts\td_replay\20260618T011656Z-v2-hard-teacher-100",
  "artifacts\td_replay\20260618T231515Z-v2-hard-laptop-700",
  "artifacts\td_replay\20260619T135628Z-v2-hard-laptop-900"
)
$hardRunRoots = @(
  "artifacts\imported\usb-20260706\artifacts\td_replay\20260618T011656Z-v2-hard-teacher-100",
  "artifacts\td_replay\20260618T231515Z-v2-hard-laptop-700",
  "artifacts\td_replay\20260619T135628Z-v2-hard-laptop-900"
)

$inputRoot = Join-Path $repoRoot "artifacts\training_inputs\two_stage_imported_20260706"
$checkpointRoot = Join-Path $repoRoot "artifacts\td_checkpoints\two_stage_imported_20260706"
$allValuePath = Join-Path $inputRoot "all.value.jsonl"
$allOpponentPath = Join-Path $inputRoot "all.opponent.jsonl"
$hardValuePath = Join-Path $inputRoot "hard-only.value.jsonl"
$hardOpponentPath = Join-Path $inputRoot "hard-only.opponent.jsonl"
$manifestPath = Join-Path $inputRoot "manifest.json"
$stage1ValueSummaryPath = Join-Path $checkpointRoot "stage1-all-value.summary.json"
$stage1OpponentSummaryPath = Join-Path $checkpointRoot "stage1-all-opponent.summary.json"
$stage2ValueSummaryPath = Join-Path $checkpointRoot "stage2-hard-only-value.summary.json"
$stage2OpponentSummaryPath = Join-Path $checkpointRoot "stage2-hard-only-opponent.summary.json"

$allInventory = Get-ReplaySetInventory -RepoRoot $repoRoot -RunRoots $allRunRoots -SetName "all"
$hardInventory = Get-ReplaySetInventory -RepoRoot $repoRoot -RunRoots $hardRunRoots -SetName "hard-only"

Write-Host "[two-stage] python=$python"
Write-Host "[two-stage] warmStartSource=$($warmStart.SourceLabel)"
Write-Host "[two-stage] warmStartRunId=$($warmStart.RunId)"
Write-Host "[two-stage] tempDir=$($cacheInfo.TempDir)"
Write-Host "[two-stage] threadCaps=OMP=$env:OMP_NUM_THREADS MKL=$env:MKL_NUM_THREADS OPENBLAS=$env:OPENBLAS_NUM_THREADS NUMEXPR=$env:NUMEXPR_NUM_THREADS"
Write-Host "[two-stage] trainThreads=$($cpuProfile.TrainThreads) trainInteropThreads=$($cpuProfile.TrainInteropThreads)"
Write-Host "[two-stage] allData games=$($allInventory.games) shards=$($allInventory.shards) rows=$($allInventory.valueRows)"
Write-Host "[two-stage] hardOnly games=$($hardInventory.games) shards=$($hardInventory.shards) rows=$($hardInventory.valueRows)"
Write-Host "[two-stage] inputRoot=$inputRoot"
Write-Host "[two-stage] checkpointRoot=$checkpointRoot"

if (-not $DryRun) {
  Join-JsonlFiles -InputPaths $allInventory.valuePaths -OutputPath $allValuePath -Force:$ForcePrepare
  Join-JsonlFiles -InputPaths $allInventory.opponentPaths -OutputPath $allOpponentPath -Force:$ForcePrepare
  Join-JsonlFiles -InputPaths $hardInventory.valuePaths -OutputPath $hardValuePath -Force:$ForcePrepare
  Join-JsonlFiles -InputPaths $hardInventory.opponentPaths -OutputPath $hardOpponentPath -Force:$ForcePrepare
  Write-TrainingInputManifest `
    -AllInventory $allInventory `
    -HardInventory $hardInventory `
    -ManifestPath $manifestPath `
    -AllValuePath $allValuePath `
    -AllOpponentPath $allOpponentPath `
    -HardValuePath $hardValuePath `
    -HardOpponentPath $hardOpponentPath
} else {
  Write-Host "[two-stage] dry run: replay concatenation skipped."
}

$commonTrainArgs = @(
  "--value-target-mode",
  "td-lambda",
  "--td-lambda",
  "0.7",
  "--save-every-steps",
  "$SaveEverySteps",
  "--progress-every-steps",
  "$ProgressEverySteps",
  "--num-threads",
  "$($cpuProfile.TrainThreads)",
  "--num-interop-threads",
  "$($cpuProfile.TrainInteropThreads)",
  "--out-dir",
  $checkpointRoot
)

$stage1ValueCommand = @(
  $python,
  "-m",
  "scripts.train_td",
  "--run-label",
  "td-two-stage-20260706-stage1-all-value",
  "--steps",
  "$Stage1Steps",
  "--seed",
  "20260706",
  "--value-replay",
  $allValuePath,
  "--warm-start-value-checkpoint",
  $warmStart.ValuePath,
  "--disable-opponent",
  "--summary-out",
  $stage1ValueSummaryPath
) + $commonTrainArgs

$stage1ValueExitCode = Invoke-MagnateLoggedCommand `
  -RepoRoot $repoRoot `
  -LogStem "train_td_two_stage_stage1_all_value" `
  -Command $stage1ValueCommand `
  -DryRun:$DryRun
if ($stage1ValueExitCode -ne 0) {
  exit $stage1ValueExitCode
}

$stage1OpponentCommand = @(
  $python,
  "-m",
  "scripts.train_td",
  "--run-label",
  "td-two-stage-20260706-stage1-all-opponent",
  "--steps",
  "$Stage1Steps",
  "--seed",
  "20260706",
  "--opponent-replay",
  $allOpponentPath,
  "--warm-start-opponent-checkpoint",
  $warmStart.OpponentPath,
  "--disable-value",
  "--summary-out",
  $stage1OpponentSummaryPath
) + $commonTrainArgs

$stage1OpponentExitCode = Invoke-MagnateLoggedCommand `
  -RepoRoot $repoRoot `
  -LogStem "train_td_two_stage_stage1_all_opponent" `
  -Command $stage1OpponentCommand `
  -DryRun:$DryRun
if ($stage1OpponentExitCode -ne 0) {
  exit $stage1OpponentExitCode
}

if ($DryRun) {
  Write-Host "[two-stage] dry run: stage 2 would warm-start from stage 1 value/opponent summaries under $checkpointRoot."
  exit 0
}

$stage1ValueCheckpoint = Get-FinalCheckpointPath -SummaryPath $stage1ValueSummaryPath -Kind "value"
$stage1OpponentCheckpoint = Get-FinalCheckpointPath -SummaryPath $stage1OpponentSummaryPath -Kind "opponent"
Write-Host "[two-stage] stage1Value=$stage1ValueCheckpoint"
Write-Host "[two-stage] stage1Opponent=$stage1OpponentCheckpoint"

$stage2ValueCommand = @(
  $python,
  "-m",
  "scripts.train_td",
  "--run-label",
  "td-two-stage-20260706-stage2-hard-only-value",
  "--steps",
  "$Stage2Steps",
  "--seed",
  "20260707",
  "--value-replay",
  $hardValuePath,
  "--warm-start-value-checkpoint",
  $stage1ValueCheckpoint,
  "--disable-opponent",
  "--summary-out",
  $stage2ValueSummaryPath
) + $commonTrainArgs

$stage2ValueExitCode = Invoke-MagnateLoggedCommand `
  -RepoRoot $repoRoot `
  -LogStem "train_td_two_stage_stage2_hard_only_value" `
  -Command $stage2ValueCommand
if ($stage2ValueExitCode -ne 0) {
  exit $stage2ValueExitCode
}

$stage2OpponentCommand = @(
  $python,
  "-m",
  "scripts.train_td",
  "--run-label",
  "td-two-stage-20260706-stage2-hard-only-opponent",
  "--steps",
  "$Stage2Steps",
  "--seed",
  "20260707",
  "--opponent-replay",
  $hardOpponentPath,
  "--warm-start-opponent-checkpoint",
  $stage1OpponentCheckpoint,
  "--disable-value",
  "--summary-out",
  $stage2OpponentSummaryPath
) + $commonTrainArgs

$stage2OpponentExitCode = Invoke-MagnateLoggedCommand `
  -RepoRoot $repoRoot `
  -LogStem "train_td_two_stage_stage2_hard_only_opponent" `
  -Command $stage2OpponentCommand
if ($stage2OpponentExitCode -ne 0) {
  exit $stage2OpponentExitCode
}

$stage2ValueCheckpoint = Get-FinalCheckpointPath -SummaryPath $stage2ValueSummaryPath -Kind "value"
$stage2OpponentCheckpoint = Get-FinalCheckpointPath -SummaryPath $stage2OpponentSummaryPath -Kind "opponent"
Write-Host "[two-stage] completed"
Write-Host "[two-stage] finalValue=$stage2ValueCheckpoint"
Write-Host "[two-stage] finalOpponent=$stage2OpponentCheckpoint"
Write-Host "[two-stage] stage1ValueSummary=$stage1ValueSummaryPath"
Write-Host "[two-stage] stage1OpponentSummary=$stage1OpponentSummaryPath"
Write-Host "[two-stage] stage2ValueSummary=$stage2ValueSummaryPath"
Write-Host "[two-stage] stage2OpponentSummary=$stage2OpponentSummaryPath"
exit 0
