function Get-MagnateRepoRoot {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptRoot
  )

  return [System.IO.Path]::GetFullPath((Join-Path $ScriptRoot ".."))
}

function Resolve-MagnateProjectPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$PathValue
  )

  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Initialize-MagnateLocalCaches {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot
  )

  $tempDir = Join-Path $RepoRoot ".tmp"
  $pipCacheDir = Join-Path $RepoRoot ".pip-cache"
  $npmCacheDir = Join-Path $RepoRoot ".npm-cache"
  $yarnCacheDir = Join-Path $RepoRoot ".yarn-cache"

  foreach ($path in @($tempDir, $pipCacheDir, $npmCacheDir, $yarnCacheDir)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
  }

  $env:TMP = $tempDir
  $env:TEMP = $tempDir
  $env:TMPDIR = $tempDir
  $env:PIP_CACHE_DIR = $pipCacheDir
  $env:NPM_CONFIG_CACHE = $npmCacheDir
  $env:YARN_CACHE_FOLDER = $yarnCacheDir
  $env:OMP_NUM_THREADS = "1"
  $env:MKL_NUM_THREADS = "1"
  $env:OPENBLAS_NUM_THREADS = "1"
  $env:NUMEXPR_NUM_THREADS = "1"
  $env:PYTHONUNBUFFERED = "1"

  return [pscustomobject]@{
    TempDir = $tempDir
    PipCacheDir = $pipCacheDir
    NpmCacheDir = $npmCacheDir
    YarnCacheDir = $yarnCacheDir
  }
}

function Get-MagnateVenvPython {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot
  )

  $pythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
  if (-not (Test-Path $pythonPath)) {
    throw "Missing project virtual environment at $pythonPath. Run .\scripts\setup_python_env.ps1 first."
  }

  return (Resolve-Path $pythonPath).Path
}

function Assert-MagnateNode22Runtime {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot
  )

  $nodePath = $null
  $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
  if ($null -ne $nodeCommand) {
    $nodePath = $nodeCommand.Source
  } else {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($env:NVM_SYMLINK)) {
      $candidates += (Join-Path $env:NVM_SYMLINK "node.exe")
    }
    $candidates += "C:\nvm4w\nodejs\node.exe"
    if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
      $candidates += (Join-Path $env:ProgramFiles "nodejs\node.exe")
    }

    foreach ($candidate in $candidates) {
      if (Test-Path $candidate) {
        $nodePath = $candidate
        break
      }
    }
  }

  if ([string]::IsNullOrWhiteSpace($nodePath)) {
    throw 'Node is not installed in a known location. Run "nvm use 22.12.0" in this shell first.'
  }

  $nodeDir = Split-Path $nodePath -Parent
  $pathEntries = @($env:Path -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($pathEntries -notcontains $nodeDir) {
    $env:Path = "$nodeDir;$env:Path"
  }

  $versionText = (& $nodePath --version).Trim()
  if ([string]::IsNullOrWhiteSpace($versionText)) {
    throw 'Unable to determine Node version. Run "nvm use 22.12.0" and try again.'
  }

  $normalizedVersion = $versionText.TrimStart("v")
  $parsedVersion = [version]$normalizedVersion
  if ($parsedVersion.Major -ne 22 -or $parsedVersion -lt [version]"22.12.0") {
    throw "Expected Node 22.12.0+ for this repo, but found $versionText. Run ""nvm use 22.12.0""."
  }

  $tsxPath = Join-Path $RepoRoot "node_modules\.bin\tsx.cmd"
  if (-not (Test-Path $tsxPath)) {
    throw 'Missing node_modules\.bin\tsx.cmd. Run "yarn install" from the repo root.'
  }

  return [pscustomobject]@{
    NodeVersion = $versionText
    TsxPath = (Resolve-Path $tsxPath).Path
  }
}

function Get-MagnateWarmStartPair {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [switch]$AllowManifestFallback
  )

  $artifactRoot = Join-Path $RepoRoot "artifacts\td_loops"
  $latest = $null

  if (Test-Path $artifactRoot) {
    foreach ($summaryPath in Get-ChildItem -Path $artifactRoot -Filter "loop.summary.json" -Recurse -File) {
      try {
        $payload = Get-Content $summaryPath.FullName -Raw | ConvertFrom-Json
      } catch {
        continue
      }

      if (-not $payload.promotion -or -not [bool]$payload.promotion.promoted) {
        continue
      }

      $chunks = @($payload.chunks)
      if ($chunks.Count -eq 0) {
        continue
      }

      $latestCheckpoint = $chunks[-1].latestCheckpoint
      if ($null -eq $latestCheckpoint) {
        continue
      }

      $valuePathRaw = [string]$latestCheckpoint.value
      $opponentPathRaw = [string]$latestCheckpoint.opponent
      if ([string]::IsNullOrWhiteSpace($valuePathRaw) -or [string]::IsNullOrWhiteSpace($opponentPathRaw)) {
        continue
      }

      $valuePath = Resolve-MagnateProjectPath -RepoRoot $RepoRoot -PathValue $valuePathRaw
      $opponentPath = Resolve-MagnateProjectPath -RepoRoot $RepoRoot -PathValue $opponentPathRaw
      if (-not (Test-Path $valuePath) -or -not (Test-Path $opponentPath)) {
        continue
      }

      $generatedAt = [DateTimeOffset]::MinValue
      $generatedAtRaw = [string]$payload.generatedAtUtc
      if (-not [string]::IsNullOrWhiteSpace($generatedAtRaw)) {
        try {
          $generatedAt = [DateTimeOffset]::Parse($generatedAtRaw)
        } catch {
          $generatedAt = [DateTimeOffset]::MinValue
        }
      }

      $candidate = [pscustomobject]@{
        Source = "artifacts"
        SourceLabel = "latest promoted loop summary"
        RunId = if ([string]::IsNullOrWhiteSpace([string]$payload.runId)) { $summaryPath.Directory.Name } else { [string]$payload.runId }
        ValuePath = $valuePath
        OpponentPath = $opponentPath
        GeneratedAt = $generatedAt
      }

      if ($null -eq $latest -or $candidate.GeneratedAt -gt $latest.GeneratedAt) {
        $latest = $candidate
      }
    }
  }

  if ($null -ne $latest) {
    return $latest
  }

  if (-not $AllowManifestFallback) {
    return $null
  }

  $manifestPath = Join-Path $RepoRoot "models\td_checkpoints\manifest.json"
  if (-not (Test-Path $manifestPath)) {
    return $null
  }

  try {
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
  } catch {
    return $null
  }

  $defaultKey = [string]$manifest.defaultWarmStart
  if ([string]::IsNullOrWhiteSpace($defaultKey)) {
    $defaultKey = "promoted"
  }

  $checkpointEntry = $manifest.checkpoints.$defaultKey
  $resolvedKey = $defaultKey
  if ($null -eq $checkpointEntry) {
    $checkpointEntry = $manifest.checkpoints.promoted
    $resolvedKey = "promoted"
  }
  if ($null -eq $checkpointEntry) {
    return $null
  }

  $valuePath = Resolve-MagnateProjectPath -RepoRoot $RepoRoot -PathValue ([string]$checkpointEntry.value)
  $opponentPath = Resolve-MagnateProjectPath -RepoRoot $RepoRoot -PathValue ([string]$checkpointEntry.opponent)
  if (-not (Test-Path $valuePath) -or -not (Test-Path $opponentPath)) {
    return $null
  }

  return [pscustomobject]@{
    Source = "manifest"
    SourceLabel = "checkpoint manifest ($resolvedKey)"
    RunId = [string]$checkpointEntry.sourceRunId
    ValuePath = $valuePath
    OpponentPath = $opponentPath
    GeneratedAt = [DateTimeOffset]::MinValue
  }
}

function Format-MagnateCommandLine {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Command
  )

  $quoted = foreach ($part in $Command) {
    if ($part -match '[\s"`]') {
      '"' + ($part -replace '"', '\"') + '"'
    } else {
      $part
    }
  }

  return ($quoted -join " ")
}

function Invoke-MagnateLoggedCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$LogStem,

    [Parameter(Mandatory = $true)]
    [string[]]$Command,

    [switch]$DryRun
  )

  $logDir = Join-Path $RepoRoot "artifacts\logs"
  New-Item -ItemType Directory -Path $logDir -Force | Out-Null

  $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmssZ")
  $logPath = Join-Path $logDir "$stamp-$LogStem.log"
  $statusPath = [System.IO.Path]::ChangeExtension($logPath, ".status")

  Write-Host "[laptop] logPath=$logPath"
  Write-Host "[laptop] statusPath=$statusPath"
  Write-Host "[laptop] command=$(Format-MagnateCommandLine -Command $Command)"

  if ($DryRun) {
    Write-Host "[laptop] dry run only; command not executed."
    return 0
  }

  $exitCode = 1
  try {
    & $Command[0] @($Command | Select-Object -Skip 1) 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
  } finally {
    $endedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    @(
      "endedAtUtc=$endedAt"
      "exitCode=$exitCode"
      "logPath=$logPath"
    ) | Set-Content -Path $statusPath

    Write-Host "[laptop] endedAtUtc=$endedAt exitCode=$exitCode"
    Write-Host "[laptop] statusPath=$statusPath"
  }

  return $exitCode
}
