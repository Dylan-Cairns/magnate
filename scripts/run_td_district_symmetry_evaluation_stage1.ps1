[CmdletBinding()]
param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-CommandArgument {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Command,

    [Parameter(Mandatory = $true)]
    [string]$Flag
  )

  $index = [Array]::IndexOf($Command, $Flag)
  if ($index -lt 0 -or $index + 1 -ge $Command.Count) {
    throw "Prepared command is missing $Flag."
  }
  return [string]$Command[$index + 1]
}

function Test-HeldoutResult {
  param(
    [Parameter(Mandatory = $true)]
    [pscustomobject]$Row
  )

  $command = [string[]]$Row.command
  $outputPath = Get-CommandArgument -Command $command -Flag "--output"
  if (-not (Test-Path $outputPath)) {
    return $false
  }

  try {
    $result = Get-Content -Raw $outputPath | ConvertFrom-Json
    $provenance = $result.provenance
    return (
      [int]$result.schemaVersion -eq 1 -and
      [int]$provenance.checkpointStep -eq 5000 -and
      [string]$provenance.valueCheckpointSha256 -eq (Get-CommandArgument -Command $command -Flag "--expected-value-checkpoint-sha256") -and
      [string]$provenance.opponentCheckpointSha256 -eq (Get-CommandArgument -Command $command -Flag "--expected-opponent-checkpoint-sha256") -and
      [string]$provenance.valueReplayContentSha256 -eq (Get-CommandArgument -Command $command -Flag "--expected-value-replay-content-sha256") -and
      [string]$provenance.opponentReplayContentSha256 -eq (Get-CommandArgument -Command $command -Flag "--expected-opponent-replay-content-sha256")
    )
  } catch {
    return $false
  }
}

function Test-SymmetryResult {
  param(
    [Parameter(Mandatory = $true)]
    [pscustomobject]$Row
  )

  $command = [string[]]$Row.command
  $outputDirectory = Get-CommandArgument -Command $command -Flag "--out-dir"
  $artifactPath = Join-Path $outputDirectory "symmetry.json"
  $summaryPath = Join-Path $outputDirectory "summary.md"
  if (-not (Test-Path $artifactPath) -or -not (Test-Path $summaryPath)) {
    return $false
  }

  try {
    $artifact = Get-Content -Raw $artifactPath | ConvertFrom-Json
    $run = $artifact.run
    return (
      [string]$artifact.artifactType -eq "ts-td-symmetry-audit" -and
      [string]$run.model.packId -eq (Get-CommandArgument -Command $command -Flag "--pack-id") -and
      [int]$run.replay.requestedSampleSize -eq [int](Get-CommandArgument -Command $command -Flag "--sample-size") -and
      [int]$run.replay.sampledRows -eq [int](Get-CommandArgument -Command $command -Flag "--sample-size") -and
      [string]$run.replay.samplingSeed -eq (Get-CommandArgument -Command $command -Flag "--sampling-seed") -and
      @($run.permutations).Count -eq 24
    )
  } catch {
    return $false
  }
}

function Invoke-PreparedCommand {
  param(
    [Parameter(Mandatory = $true)]
    [pscustomobject]$Row,

    [Parameter(Mandatory = $true)]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [int]$Index,

    [Parameter(Mandatory = $true)]
    [int]$Count,

    [Parameter(Mandatory = $true)]
    [string]$LogRoot
  )

  $command = [string[]]$Row.command
  $logPath = Join-Path $LogRoot "$Stage-$($Row.id).log"
  Write-Host "[evaluation-stage1] START $Stage $Index/$Count $($Row.id) at $(Get-Date -Format o)" -ForegroundColor Cyan

  $executable = $command[0]
  $arguments = [string[]]($command | Select-Object -Skip 1)
  & $executable @arguments 2>&1 | Tee-Object -FilePath $logPath
  $exitCode = [int]$LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "Prepared command $($Row.id) failed with exit code $exitCode. See $logPath"
  }

  Write-Host "[evaluation-stage1] COMPLETE $Stage $Index/$Count $($Row.id) at $(Get-Date -Format o)" -ForegroundColor Green
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$planPath = Join-Path $repoRoot "artifacts\training_inputs\district-s4-ablation-pilot-v1\post-training-evaluation\evaluation-plan.json"
$artifactRoot = Join-Path $repoRoot "artifacts\td_ablation_evals\district-s4-ablation-pilot-v1"
$logRoot = Join-Path $artifactRoot "launch-logs"
$completionPath = Join-Path $logRoot "evaluation-stage1.complete"
$deployedIndexPath = Join-Path $repoRoot "public\model-packs\index.json"
$checkpointRegistryPath = Join-Path $repoRoot "models\td_checkpoints\manifest.json"

if (-not (Test-Path $planPath)) {
  throw "Missing prepared evaluation plan at $planPath. Run .\.venv\Scripts\python.exe -m scripts.prepare_td_district_symmetry_evaluation first."
}

$plan = Get-Content -Raw $planPath | ConvertFrom-Json
$exports = @($plan.exportCommands)
$heldout = @($plan.heldoutCommands)
$symmetry = @($plan.symmetryCommands)
if ([string]$plan.status -ne "prepared-not-launched" -or [bool]$plan.launchAuthorized) {
  throw "Evaluation plan is not in the expected prepared-not-launched state."
}
if ($exports.Count -ne 7 -or $heldout.Count -ne 4 -or $symmetry.Count -ne 4) {
  throw "Expected 7 exports, 4 held-out commands, and 4 symmetry commands; found $($exports.Count), $($heldout.Count), and $($symmetry.Count)."
}

foreach ($row in $exports) {
  $command = [string[]]$row.command
  if (
    $command -notcontains "--no-set-default" -or
    (Get-CommandArgument -Command $command -Flag "--output-root") -notmatch 'public\\model-packs-experiments\\district-s4-ablation-pilot-v1$'
  ) {
    throw "Export command $($row.id) is not isolated from the deployed model-pack index."
  }
}

foreach ($row in $heldout) {
  $command = [string[]]$row.command
  if ((Get-CommandArgument -Command $command -Flag "--num-threads") -ne "4") {
    throw "Held-out command $($row.id) is not capped at four threads."
  }
  if ((Get-CommandArgument -Command $command -Flag "--expected-checkpoint-step") -ne "5000") {
    throw "Held-out command $($row.id) is not pinned to checkpoint step 5,000."
  }
}

if (-not $DryRun) {
  $activeEvaluation = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python(\.exe)?$' -and $_.CommandLine -match 'evaluate_td_replay_holdout|export_browser_td_root_pack') -or
    ($_.Name -match '^(node|yarn)(\.exe|\.cmd)?$' -and $_.CommandLine -match 'td-symmetry')
  }
  if ($null -ne $activeEvaluation) {
    $processIds = @($activeEvaluation | ForEach-Object ProcessId) -join ", "
    throw "A stage-one evaluation process is already running (process IDs: $processIds). Refusing to launch a duplicate."
  }
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
if ((Test-Path $completionPath) -and -not $DryRun) {
  Remove-Item -LiteralPath $completionPath -Force
}

$deployedIndexSha256Before = (Get-FileHash -Algorithm SHA256 $deployedIndexPath).Hash.ToLowerInvariant()
$checkpointRegistrySha256Before = (Get-FileHash -Algorithm SHA256 $checkpointRegistryPath).Hash.ToLowerInvariant()
$planSha256 = (Get-FileHash -Algorithm SHA256 $planPath).Hash.ToLowerInvariant()

Write-Host "[evaluation-stage1] plan=$planPath"
Write-Host "[evaluation-stage1] planSha256=$planSha256"
Write-Host "[evaluation-stage1] execution=sequential"
Write-Host "[evaluation-stage1] stages=exports,heldout,symmetry"
Write-Host "[evaluation-stage1] logs=$logRoot"
if ($DryRun) {
  Write-Host "[evaluation-stage1] dry run only; no exports or evaluations will be started." -ForegroundColor Yellow
}

for ($index = 0; $index -lt $exports.Count; $index++) {
  $row = $exports[$index]
  if ($DryRun) {
    Write-Host "[evaluation-stage1] WOULD RUN export $($index + 1)/$($exports.Count) $($row.id)"
    continue
  }

  Invoke-PreparedCommand -Row $row -Stage "export" -Index ($index + 1) -Count $exports.Count -LogRoot $logRoot

  $command = [string[]]$row.command
  $outputRoot = Get-CommandArgument -Command $command -Flag "--output-root"
  $packId = Get-CommandArgument -Command $command -Flag "--pack-id"
  $packManifestPath = Join-Path (Join-Path $outputRoot $packId) "manifest.json"
  $weightsPath = Join-Path (Join-Path $outputRoot $packId) "weights.json"
  if (-not (Test-Path $packManifestPath) -or -not (Test-Path $weightsPath)) {
    throw "Export $($row.id) did not produce its expected manifest and weights."
  }
  $packManifest = Get-Content -Raw $packManifestPath | ConvertFrom-Json
  if (
    [string]$packManifest.packId -ne $packId -or
    [string]$packManifest.source.valueCheckpoint -ne (Get-CommandArgument -Command $command -Flag "--value-checkpoint") -or
    [string]$packManifest.source.opponentCheckpoint -ne (Get-CommandArgument -Command $command -Flag "--opponent-checkpoint")
  ) {
    throw "Exported pack $packId does not match its frozen checkpoint command."
  }
}

if (-not $DryRun) {
  $firstExportCommand = [string[]]$exports[0].command
  $experimentalRoot = Get-CommandArgument -Command $firstExportCommand -Flag "--output-root"
  $experimentalIndexPath = Join-Path $experimentalRoot "index.json"
  if (-not (Test-Path $experimentalIndexPath)) {
    throw "Experimental model-pack index was not created: $experimentalIndexPath"
  }
  $experimentalIndex = Get-Content -Raw $experimentalIndexPath | ConvertFrom-Json
  $exportedPackIds = @($experimentalIndex.packs | ForEach-Object { [string]$_.id })
  foreach ($row in $exports) {
    if ($exportedPackIds -notcontains [string]$row.packId) {
      throw "Experimental model-pack index is missing $($row.packId)."
    }
  }
  if (-not [string]::IsNullOrWhiteSpace([string]$experimentalIndex.defaultPackId)) {
    throw "Experimental model-pack index unexpectedly selected a default pack."
  }
}

for ($index = 0; $index -lt $heldout.Count; $index++) {
  $row = $heldout[$index]
  if (Test-HeldoutResult -Row $row) {
    Write-Host "[evaluation-stage1] SKIP heldout $($index + 1)/$($heldout.Count) $($row.id) (validated result exists)" -ForegroundColor Yellow
    continue
  }
  if ($DryRun) {
    Write-Host "[evaluation-stage1] WOULD RUN heldout $($index + 1)/$($heldout.Count) $($row.id)"
    continue
  }

  Invoke-PreparedCommand -Row $row -Stage "heldout" -Index ($index + 1) -Count $heldout.Count -LogRoot $logRoot
  if (-not (Test-HeldoutResult -Row $row)) {
    throw "Held-out command $($row.id) completed but its result failed validation."
  }
}

for ($index = 0; $index -lt $symmetry.Count; $index++) {
  $row = $symmetry[$index]
  if (Test-SymmetryResult -Row $row) {
    Write-Host "[evaluation-stage1] SKIP symmetry $($index + 1)/$($symmetry.Count) $($row.id) (validated result exists)" -ForegroundColor Yellow
    continue
  }
  if ($DryRun) {
    Write-Host "[evaluation-stage1] WOULD RUN symmetry $($index + 1)/$($symmetry.Count) $($row.id)"
    continue
  }

  Invoke-PreparedCommand -Row $row -Stage "symmetry" -Index ($index + 1) -Count $symmetry.Count -LogRoot $logRoot
  if (-not (Test-SymmetryResult -Row $row)) {
    throw "Symmetry command $($row.id) completed but its result failed validation."
  }
}

if ($DryRun) {
  Write-Host "[evaluation-stage1] DRY RUN COMPLETE" -ForegroundColor Green
  exit 0
}

$deployedIndexSha256After = (Get-FileHash -Algorithm SHA256 $deployedIndexPath).Hash.ToLowerInvariant()
$checkpointRegistrySha256After = (Get-FileHash -Algorithm SHA256 $checkpointRegistryPath).Hash.ToLowerInvariant()
if ($deployedIndexSha256After -ne $deployedIndexSha256Before) {
  throw "Deployed model-pack index changed during stage one."
}
if ($checkpointRegistrySha256After -ne $checkpointRegistrySha256Before) {
  throw "Checkpoint registry changed during stage one."
}

@(
  "completedAtUtc=$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
  "evaluationPlanSha256=$planSha256"
  "exports=7"
  "heldout=4"
  "symmetry=4"
  "execution=sequential"
  "heldoutThreads=4"
  "deployedIndexSha256=$deployedIndexSha256After"
  "checkpointRegistrySha256=$checkpointRegistrySha256After"
) | Set-Content -Path $completionPath -Encoding utf8

Write-Host ""
Write-Host "FIRST-STAGE DISTRICT-SYMMETRY EVALUATION IS COMPLETE" -ForegroundColor Green
Write-Host "[evaluation-stage1] completionMarker=$completionPath"
