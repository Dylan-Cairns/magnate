[CmdletBinding()]
param(
    [ValidateSet('smoke', 'full')]
    [string]$Mode = 'full',
    [int]$States = 0,
    [int]$Rounds = 0,
    [int]$WarmupRounds = 0,
    [double]$MinimumSpeedup = 1.3,
    [int]$TimeoutMinutes = 0,
    [int]$Port = 4187,
    [string]$BrowserPath = '',
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($States -le 0) {
    $States = if ($Mode -eq 'smoke') { 8 } else { 128 }
}
if ($Rounds -le 0) {
    $Rounds = if ($Mode -eq 'smoke') { 1 } else { 500 }
}
if ($WarmupRounds -le 0) {
    $WarmupRounds = if ($Mode -eq 'smoke') { 1 } else { 5 }
}
if ($States % 2 -ne 0) {
    throw '-States must be even.'
}
if ($MinimumSpeedup -le 0) {
    throw '-MinimumSpeedup must be greater than zero.'
}
if (-not $PSBoundParameters.ContainsKey('TimeoutMinutes')) {
    $TimeoutMinutes = if ($Mode -eq 'smoke') { 2 } else { 0 }
}
if ($TimeoutMinutes -lt 0) {
    throw '-TimeoutMinutes must be zero (disabled) or a positive integer.'
}
if ($Port -le 0 -or $Port -ge 65534) {
    throw '-Port must leave room for an adjacent browser debugging port.'
}
$debugPort = $Port + 1

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $timestamp = Get-Date -Format 'yyyyMMddTHHmmssZ'
    $OutDir = Join-Path $repoRoot "artifacts/benchmarks/td-two-lane-browser-$timestamp"
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repoRoot $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
[void](New-Item -ItemType Directory -Path $OutDir -Force)

$resolvedBrowser = $BrowserPath
if ([string]::IsNullOrWhiteSpace($resolvedBrowser)) {
    $commandCandidates = @('msedge.exe', 'chrome.exe')
    foreach ($candidate in $commandCandidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            $resolvedBrowser = $command.Source
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($resolvedBrowser)) {
    $pathCandidates = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    $resolvedBrowser = $pathCandidates |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_) } |
        Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($resolvedBrowser) -or -not (Test-Path -LiteralPath $resolvedBrowser)) {
    throw 'Could not find Edge or Chrome. Pass -BrowserPath with an explicit executable path.'
}
$resolvedBrowser = (Resolve-Path -LiteralPath $resolvedBrowser).Path

$nodeCommand = Get-Command node -ErrorAction Stop
$nodeVersion = & $nodeCommand.Source -p 'process.versions.node'
if (-not $nodeVersion.StartsWith('22.', [StringComparison]::Ordinal)) {
    $nvmRoot = Join-Path $env:APPDATA 'nvm'
    $node22Candidates = Get-ChildItem -LiteralPath $nvmRoot -Directory -Filter 'v22.*' -ErrorAction SilentlyContinue |
        Sort-Object { [version]$_.Name.Substring(1) } -Descending |
        ForEach-Object { Join-Path $_.FullName 'node.exe' } |
        Where-Object { Test-Path -LiteralPath $_ }
    $node22Path = $node22Candidates | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($node22Path)) {
        throw "The benchmark requires Node 22. Current Node is $nodeVersion and no Node 22 nvm installation was found."
    }
    $nodeCommand = Get-Item -LiteralPath $node22Path
    $nodeVersion = & $nodeCommand.FullName -p 'process.versions.node'
}
$nodeExecutable = if (-not [string]::IsNullOrWhiteSpace($nodeCommand.Source)) {
    $nodeCommand.Source
} else {
    $nodeCommand.FullName
}
Write-Host "Using Node $nodeVersion from $nodeExecutable"
$viteEntry = Join-Path $repoRoot 'node_modules/vite/bin/vite.js'
if (-not (Test-Path -LiteralPath $viteEntry)) {
    throw 'Vite is not installed. Run npm install before the benchmark.'
}

$viteStdout = Join-Path $OutDir 'vite.stdout.log'
$viteStderr = Join-Path $OutDir 'vite.stderr.log'
$browserStdout = Join-Path $OutDir 'browser.stdout.log'
$browserStderr = Join-Path $OutDir 'browser.stderr.log'
$envelopePath = Join-Path $OutDir 'browser-result-envelope.json'
$jsonPath = Join-Path $OutDir 'result.json'
$browserProfile = Join-Path ([IO.Path]::GetTempPath()) "magnate-td-two-lane-$PID-$([guid]::NewGuid().ToString('N'))"
[void](New-Item -ItemType Directory -Path $browserProfile)

$viteProcess = $null
$browserProcess = $null
try {
    $viteArguments = @{
        FilePath = $nodeExecutable
        ArgumentList = @($viteEntry, '--host', '127.0.0.1', '--port', "$Port", '--strictPort')
        WorkingDirectory = $repoRoot
        WindowStyle = 'Hidden'
        RedirectStandardOutput = $viteStdout
        RedirectStandardError = $viteStderr
        PassThru = $true
    }
    $viteProcess = Start-Process @viteArguments

    $benchmarkBaseUrl = "http://127.0.0.1:$Port/benchmarks/td-two-lane.html"
    $serverReady = $false
    for ($attempt = 0; $attempt -lt 80; $attempt += 1) {
        if ($viteProcess.HasExited) {
            throw "Vite exited before the benchmark started. See $viteStderr"
        }
        try {
            $response = Invoke-WebRequest -Uri $benchmarkBaseUrl -TimeoutSec 2 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                $serverReady = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $serverReady) {
        throw "Timed out waiting for Vite. See $viteStderr"
    }

    $query = @(
        "mode=$Mode",
        "states=$States",
        "rounds=$Rounds",
        "warmupRounds=$WarmupRounds",
        "minimumSpeedup=$MinimumSpeedup"
    ) -join '&'
    $benchmarkUrl = "$benchmarkBaseUrl`?$query"
    Write-Host "Running TD two-lane browser benchmark in $resolvedBrowser"
    Write-Host "URL: $benchmarkUrl"

    $browserArguments = @(
        '--headless=new',
        '--disable-gpu',
        '--disable-background-networking',
        '--disable-default-apps',
        '--no-first-run',
        "--remote-debugging-port=$debugPort",
        '--remote-allow-origins=*',
        "--user-data-dir=$browserProfile",
        $benchmarkUrl
    )
    $browserLaunch = @{
        FilePath = $resolvedBrowser
        ArgumentList = $browserArguments
        WindowStyle = 'Hidden'
        RedirectStandardOutput = $browserStdout
        RedirectStandardError = $browserStderr
        PassThru = $true
    }
    $browserProcess = Start-Process @browserLaunch
    $waitHelper = Join-Path $repoRoot 'scripts/wait_for_td_two_lane_browser_result.mjs'
    $timeoutMs = $TimeoutMinutes * 60 * 1000
    & $nodeExecutable $waitHelper '--port' "$debugPort" '--timeout-ms' "$timeoutMs" '--out' $envelopePath
    if ($LASTEXITCODE -ne 0) {
        throw "Browser result waiter exited with code $LASTEXITCODE. See $browserStderr"
    }
    $envelope = Get-Content -LiteralPath $envelopePath -Raw | ConvertFrom-Json
    $status = $envelope.status
    $payload = $envelope.payload
    [IO.File]::WriteAllText($jsonPath, $payload)
    if ($status -ne 'complete') {
        throw "Browser benchmark failed. See $jsonPath"
    }

    $result = $payload | ConvertFrom-Json
    Write-Host "Result: $jsonPath"
    Write-Host ("Exact mismatches: {0}" -f $result.correctness.exactMismatchCount)
    Write-Host ("Argmax mismatches: {0}" -f $result.correctness.argmaxMismatchCount)
    Write-Host ("Two-lane speedup: {0:N3}x" -f $result.timing.speedup)
    Write-Host ("Recommendation: {0}" -f $result.gate.recommendation)
} finally {
    if ($null -ne $browserProcess -and -not $browserProcess.HasExited) {
        Stop-Process -Id $browserProcess.Id -Force
        $browserProcess.WaitForExit()
    }
    if ($null -ne $viteProcess -and -not $viteProcess.HasExited) {
        Stop-Process -Id $viteProcess.Id -Force
        $viteProcess.WaitForExit()
    }
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $resolvedProfile = [IO.Path]::GetFullPath($browserProfile)
    if (
        $resolvedProfile.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        [IO.Path]::GetFileName($resolvedProfile).StartsWith('magnate-td-two-lane-', [StringComparison]::Ordinal)
    ) {
        Remove-Item -LiteralPath $resolvedProfile -Recurse -Force -ErrorAction SilentlyContinue
    }
}
