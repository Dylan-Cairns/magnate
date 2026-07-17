[CmdletBinding()]
param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$manifestPath = Join-Path $repoRoot "artifacts\training_inputs\district-s4-opponent-orbit-pilot-v1\resolved-manifest.json"
$logRoot = Join-Path $repoRoot "artifacts\td_checkpoints\district-s4-opponent-orbit-pilot-v1\launch-logs"
$completionPath = Join-Path $logRoot "pilot-training.complete"

function Get-CommandFlagValue {
  param(
    [Parameter(Mandatory = $true)][string[]]$Command,
    [Parameter(Mandatory = $true)][string]$Flag
  )
  $index = [Array]::IndexOf($Command, $Flag)
  if ($index -lt 0 -or $index + 1 -ge $Command.Count) {
    throw "Frozen command is missing $Flag or its value."
  }
  return [string]$Command[$index + 1]
}

function Test-CompletedSummary {
  param(
    [Parameter(Mandatory = $true)]$Row,
    [Parameter(Mandatory = $true)]$Manifest
  )
  $command = [string[]]$Row.command
  $summaryPath = Get-CommandFlagValue -Command $command -Flag "--summary-out"
  if (-not (Test-Path -LiteralPath $summaryPath)) {
    return $false
  }

  try {
    $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    $expectedMode = switch ([string]$Row.districtAugmentation) {
      "s4" { "pawn-district-s4-d3-fixed-v1" }
      "s4-orbit" { "pawn-district-s4-orbit-d3-fixed-v1" }
      default { throw "Unsupported frozen augmentation mode: $($Row.districtAugmentation)" }
    }
    $expectedSeed = [int](Get-CommandFlagValue -Command $command -Flag "--seed")
    $expectedManifestSha = [string]$Manifest.sourceManifestSha256
    $expectedImplementationSha = [string]$Manifest.verification.implementationSha256
    $expectedReplaySha = [string]$Manifest.verification.contentSha256.training.opponent
    $expectedWarmStartSha = [string]$Manifest.verification.opponentWarmStart.sha256

    if ([int]$summary.config.steps -ne 5000) { throw "summary steps mismatch" }
    if ([int]$summary.config.seed -ne $expectedSeed) { throw "summary seed mismatch" }
    if ([bool]$summary.config.trainValue) { throw "summary unexpectedly trained value" }
    if (-not [bool]$summary.config.trainOpponent) { throw "summary did not train opponent" }
    if ([string]$summary.config.districtAugmentation -ne $expectedMode) {
      throw "summary augmentation mode mismatch"
    }
    if ([int]$summary.config.opponentAugmentationCopiesPerRawSample -ne [int]$Row.copiesPerRawSample) {
      throw "summary augmentation multiplicity mismatch"
    }
    if ([int]$summary.config.opponentEffectiveBatchSize -ne (64 * [int]$Row.copiesPerRawSample)) {
      throw "summary effective batch size mismatch"
    }
    if ([int]$summary.config.numThreads -ne 4 -or [int]$summary.config.numInteropThreads -ne 1) {
      throw "summary thread profile mismatch"
    }
    if ([int]$summary.results.opponentReplaySize -ne 145014) {
      throw "summary replay row count mismatch"
    }
    if ([string]$summary.provenance.experimentManifestSha256 -ne $expectedManifestSha) {
      throw "summary source-manifest hash mismatch"
    }
    if ([string]$summary.provenance.implementationSha256 -ne $expectedImplementationSha) {
      throw "summary implementation hash mismatch"
    }
    if ([string]$summary.provenance.opponentReplayContentSha256 -ne $expectedReplaySha) {
      throw "summary replay-content hash mismatch"
    }
    if ([string]$summary.provenance.warmStartOpponentSha256 -ne $expectedWarmStartSha) {
      throw "summary warm-start hash mismatch"
    }
    if ([string]$summary.results.opponentSamplingTraceSha256 -notmatch '^[0-9a-f]{64}$') {
      throw "summary sampling trace is missing or malformed"
    }

    $finalCheckpoint = @($summary.results.checkpoints) |
      Where-Object { [int]$_.step -eq 5000 } |
      Select-Object -Last 1
    if ($null -eq $finalCheckpoint -or -not (Test-Path -LiteralPath ([string]$finalCheckpoint.opponent))) {
      throw "final opponent checkpoint is missing"
    }
  } catch {
    throw "Existing summary for $($Row.id) is not a valid completed run: $summaryPath ($($_.Exception.Message))"
  }
  return $true
}

if (-not (Test-Path -LiteralPath $python)) {
  throw "Missing project Python runtime at $python"
}

Push-Location $repoRoot
try {
  Write-Host "[orbit-pilot] validating and resolving frozen inputs"
  & $python -m scripts.prepare_td_opponent_orbit_ablation
  if ($LASTEXITCODE -ne 0) {
    throw "Opponent-orbit pilot preparation failed with exit code $LASTEXITCODE."
  }

  if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Preparation did not write the resolved manifest at $manifestPath"
  }
  $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
  $commands = @($manifest.commands)
  if ($commands.Count -ne 4) {
    throw "Expected exactly four frozen opponent-only commands, but found $($commands.Count)."
  }
  if ([bool]$manifest.launchAuthorized) {
    throw "Resolved manifest unexpectedly has launchAuthorized=true."
  }

  $expectedArms = @{
    "random-s4-control" = @{ mode = "s4"; copies = 1 }
    "complete-s4-orbit" = @{ mode = "s4-orbit"; copies = 24 }
  }
  foreach ($row in $commands) {
    $command = [string[]]$row.command
    foreach ($requiredFlag in @(
      "--steps",
      "--opponent-batch-size",
      "--district-augmentation",
      "--num-threads",
      "--num-interop-threads",
      "--summary-out"
    )) {
      $null = Get-CommandFlagValue -Command $command -Flag $requiredFlag
    }
    if ([Array]::IndexOf($command, "--disable-value") -lt 0) {
      throw "Frozen command $($row.id) is not opponent-only."
    }
    $arm = $expectedArms[[string]$row.armId]
    if ($null -eq $arm) { throw "Unexpected pilot arm: $($row.armId)" }
    if ((Get-CommandFlagValue $command "--steps") -ne "5000") {
      throw "Frozen command $($row.id) is not a 5,000-update run."
    }
    if ((Get-CommandFlagValue $command "--opponent-batch-size") -ne "64") {
      throw "Frozen command $($row.id) does not use raw batch size 64."
    }
    if ((Get-CommandFlagValue $command "--num-threads") -ne "4" -or
        (Get-CommandFlagValue $command "--num-interop-threads") -ne "1") {
      throw "Frozen command $($row.id) does not use the approved four-thread profile."
    }
    if ((Get-CommandFlagValue $command "--district-augmentation") -ne [string]$arm.mode -or
        [int]$row.copiesPerRawSample -ne [int]$arm.copies) {
      throw "Frozen command $($row.id) does not match its arm's augmentation contract."
    }
  }

  if (-not $DryRun) {
    $activePilot = Get-CimInstance Win32_Process | Where-Object {
      $_.Name -match '^python(\.exe)?$' -and
      $_.CommandLine -match 'scripts\.train_td' -and
      $_.CommandLine -match 'district-s4-opponent-orbit-pilot-v1'
    }
    if ($null -ne $activePilot) {
      $processIds = @($activePilot | ForEach-Object ProcessId) -join ", "
      throw "An opponent-orbit pilot trainer is already running (process IDs: $processIds)."
    }
  }

  $env:OMP_NUM_THREADS = "4"
  $env:MKL_NUM_THREADS = "4"
  $env:OPENBLAS_NUM_THREADS = "4"
  $env:NUMEXPR_NUM_THREADS = "4"

  New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
  Write-Host "[orbit-pilot] execution=sequential commands=4"
  Write-Host "[orbit-pilot] trainerThreads=4 interopThreads=1"
  Write-Host "[orbit-pilot] rawBatch=64 orbitEffectiveBatch=1536"
  Write-Host "[orbit-pilot] logs=$logRoot"
  if ($DryRun) {
    Write-Host "[orbit-pilot] dry run only; no training will be started." -ForegroundColor Yellow
  }

  for ($index = 0; $index -lt $commands.Count; $index++) {
    $row = $commands[$index]
    if (Test-CompletedSummary -Row $row -Manifest $manifest) {
      Write-Host "[orbit-pilot] SKIP $($index + 1)/4: $($row.id) (validated summary exists)" -ForegroundColor Yellow
      continue
    }
    if ($DryRun) {
      Write-Host "[orbit-pilot] WOULD RUN $($index + 1)/4: $($row.id)"
      continue
    }

    $command = [string[]]$row.command
    $summaryPath = Get-CommandFlagValue -Command $command -Flag "--summary-out"
    $logPath = Join-Path $logRoot "$($row.id).log"
    Write-Host "[orbit-pilot] START $($index + 1)/4: $($row.id) at $(Get-Date -Format o)" -ForegroundColor Cyan
    $executable = $command[0]
    $arguments = [string[]]($command | Select-Object -Skip 1)
    & $executable @arguments 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = [int]$LASTEXITCODE
    if ($exitCode -ne 0) {
      throw "Pilot command $($row.id) failed with exit code $exitCode. See $logPath"
    }
    if (-not (Test-Path -LiteralPath $summaryPath)) {
      throw "Pilot command $($row.id) did not write its expected summary: $summaryPath"
    }
    $null = Test-CompletedSummary -Row $row -Manifest $manifest
    Write-Host "[orbit-pilot] COMPLETE $($index + 1)/4: $($row.id) at $(Get-Date -Format o)" -ForegroundColor Green
  }

  if ($DryRun) {
    Write-Host "[orbit-pilot] DRY RUN COMPLETE" -ForegroundColor Green
    exit 0
  }

  foreach ($seedId in @("pilot-a", "pilot-b")) {
    $seedRows = @($commands | Where-Object { [string]$_.seedId -eq $seedId })
    if ($seedRows.Count -ne 2) { throw "Expected two matched summaries for $seedId." }
    $traces = @()
    foreach ($row in $seedRows) {
      $command = [string[]]$row.command
      $summaryPath = Get-CommandFlagValue -Command $command -Flag "--summary-out"
      $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
      $traces += [string]$summary.results.opponentSamplingTraceSha256
    }
    if ($traces[0] -ne $traces[1]) {
      throw "Matched raw sampling traces differ for $seedId."
    }
    Write-Host "[orbit-pilot] MATCHED $seedId rawSamplingTraceSha256=$($traces[0])" -ForegroundColor Green
  }

  @(
    "completedAtUtc=$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    "commands=4"
    "stepsPerCommand=5000"
    "trainerThreads=4"
    "interopThreads=1"
  ) | Set-Content -LiteralPath $completionPath -Encoding utf8

  Write-Host ""
  Write-Host "ALL FOUR OPPONENT-ORBIT PILOT JOBS ARE COMPLETE" -ForegroundColor Green
  Write-Host "[orbit-pilot] completionMarker=$completionPath"
} finally {
  Pop-Location
}
