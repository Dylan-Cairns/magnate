[CmdletBinding()]
param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$manifestPath = Join-Path $repoRoot "artifacts\training_inputs\district-s4-ablation-pilot-v1\resolved-manifest.json"
$logRoot = Join-Path $repoRoot "artifacts\td_checkpoints\district-s4-ablation-pilot-v1\launch-logs"
$completionPath = Join-Path $logRoot "pilot-training.complete"

if (-not (Test-Path $manifestPath)) {
  throw "Missing resolved pilot manifest at $manifestPath. Run .\.venv\Scripts\python.exe -m scripts.prepare_td_district_symmetry_ablation first."
}

$manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
$commands = @($manifest.commands)
if ($commands.Count -ne 8) {
  throw "Expected exactly eight frozen pilot commands, but found $($commands.Count)."
}
if ([bool]$manifest.launchAuthorized) {
  throw "Resolved manifest unexpectedly has launchAuthorized=true; the frozen preparation contract requires false."
}

foreach ($row in $commands) {
  $command = [string[]]$row.command
  foreach ($requiredFlag in @("--steps", "--num-threads", "--num-interop-threads", "--summary-out")) {
    if ([Array]::IndexOf($command, $requiredFlag) -lt 0) {
      throw "Frozen command $($row.id) is missing $requiredFlag."
    }
  }

  $steps = $command[[Array]::IndexOf($command, "--steps") + 1]
  $threads = $command[[Array]::IndexOf($command, "--num-threads") + 1]
  $interopThreads = $command[[Array]::IndexOf($command, "--num-interop-threads") + 1]
  if ($steps -ne "5000" -or $threads -ne "4" -or $interopThreads -ne "1") {
    throw "Frozen command $($row.id) does not match the approved 5,000-update, four-thread pilot profile."
  }
}

if (-not $DryRun) {
  $activePilot = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -match 'scripts\.train_td' -and
    $_.CommandLine -match 'district-s4-ablation-pilot-v1'
  }
  if ($null -ne $activePilot) {
    $processIds = @($activePilot | ForEach-Object ProcessId) -join ", "
    throw "A district-S4 pilot trainer is already running (process IDs: $processIds). Refusing to launch a duplicate."
  }
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
if ((Test-Path $completionPath) -and -not $DryRun) {
  Remove-Item -LiteralPath $completionPath -Force
}

Write-Host "[pilot] manifest=$manifestPath"
Write-Host "[pilot] execution=sequential"
Write-Host "[pilot] trainerThreads=4 interopThreads=1"
Write-Host "[pilot] logs=$logRoot"
if ($DryRun) {
  Write-Host "[pilot] dry run only; no training will be started." -ForegroundColor Yellow
}

for ($index = 0; $index -lt $commands.Count; $index++) {
  $row = $commands[$index]
  $command = [string[]]$row.command
  $summaryPath = $command[[Array]::IndexOf($command, "--summary-out") + 1]

  if (Test-Path $summaryPath) {
    try {
      $summary = Get-Content -Raw $summaryPath | ConvertFrom-Json
      if ([int]$summary.config.steps -ne 5000) {
        throw "summary reports $($summary.config.steps) steps"
      }
    } catch {
      throw "Existing summary for $($row.id) is not a valid completed 5,000-update summary: $summaryPath ($($_.Exception.Message))"
    }

    Write-Host "[pilot] SKIP $($index + 1)/$($commands.Count): $($row.id) (validated summary exists)" -ForegroundColor Yellow
    continue
  }

  if ($DryRun) {
    Write-Host "[pilot] WOULD RUN $($index + 1)/$($commands.Count): $($row.id)"
    continue
  }

  $logPath = Join-Path $logRoot "$($row.id).log"
  Write-Host "[pilot] START $($index + 1)/$($commands.Count): $($row.id) at $(Get-Date -Format o)" -ForegroundColor Cyan

  $executable = $command[0]
  $arguments = [string[]]($command | Select-Object -Skip 1)
  & $executable @arguments 2>&1 | Tee-Object -FilePath $logPath
  $exitCode = [int]$LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "Pilot command $($row.id) failed with exit code $exitCode. See $logPath"
  }
  if (-not (Test-Path $summaryPath)) {
    throw "Pilot command $($row.id) exited successfully but did not write its expected summary: $summaryPath"
  }

  Write-Host "[pilot] COMPLETE $($index + 1)/$($commands.Count): $($row.id) at $(Get-Date -Format o)" -ForegroundColor Green
}

if ($DryRun) {
  Write-Host "[pilot] DRY RUN COMPLETE" -ForegroundColor Green
  exit 0
}

$missingSummaries = @()
foreach ($row in $commands) {
  $command = [string[]]$row.command
  $summaryPath = $command[[Array]::IndexOf($command, "--summary-out") + 1]
  if (-not (Test-Path $summaryPath)) {
    $missingSummaries += $summaryPath
  }
}
if ($missingSummaries.Count -gt 0) {
  throw "Training loop ended with missing summaries: $($missingSummaries -join ', ')"
}

@(
  "completedAtUtc=$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
  "commands=8"
  "stepsPerCommand=5000"
  "trainerThreads=4"
  "interopThreads=1"
) | Set-Content -Path $completionPath -Encoding utf8

Write-Host ""
Write-Host "ALL EIGHT PILOT TRAINING JOBS ARE COMPLETE" -ForegroundColor Green
Write-Host "[pilot] completionMarker=$completionPath"
