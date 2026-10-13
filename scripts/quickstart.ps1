# qwen35 harness - one-shot setup and smoke test (Windows PowerShell)
#
#   pwsh -File scripts\quickstart.ps1
#   pwsh -File scripts\quickstart.ps1 -ModelDir D:\qwen35-models -Workspace .\myproject
#
# Safe to re-run: downloads are skipped when already present.

[CmdletBinding()]
param(
    [string]$ModelDir = 'D:\qwen35-models',
    [string]$Workspace = '',
    [int]$Context = 32768,
    [switch]$SkipServe,
    [switch]$SkipModel
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Find-Python {
    foreach ($candidate in @('python', 'py')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    throw 'No python interpreter found on PATH.'
}

$python = Find-Python
Write-Host "harness   : $root" -ForegroundColor Cyan
Write-Host "interpreter: $python" -ForegroundColor Cyan

# ---------------------------------------------------------------- environment
Write-Host "`n=== environment ===" -ForegroundColor Cyan
& $python -m qwen35 doctor --model-dir $ModelDir

# ---------------------------------------------------------------- llama.cpp
Write-Host "`n=== llama.cpp runtime ===" -ForegroundColor Cyan
& $python -m qwen35 install-runtime
if ($LASTEXITCODE -ne 0) { throw 'could not install the llama.cpp runtime' }

# ---------------------------------------------------------------- weights
if (-not $SkipModel) {
    Write-Host "`n=== weights ===" -ForegroundColor Cyan
    & $python -m qwen35 fetch --model-dir $ModelDir
    if ($LASTEXITCODE -ne 0) { throw 'could not download the model weights' }
}

# ---------------------------------------------------------------- self-tests
Write-Host "`n=== self-tests (no model required) ===" -ForegroundColor Cyan
& $python -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw 'self-tests failed' }

Write-Host "`n=== workspace confinement ===" -ForegroundColor Cyan
& $python -m qwen35 workspaces

# ---------------------------------------------------------------- server + demo
if (-not $SkipServe) {
    Write-Host "`n=== starting llama-server ===" -ForegroundColor Cyan
    $serve = Start-Process -FilePath $python `
        -ArgumentList @('-m', 'qwen35', 'serve', '--model-dir', $ModelDir, '--context', $Context) `
        -WorkingDirectory $root -PassThru -WindowStyle Hidden

    Write-Host "server pid $($serve.Id); waiting for it to become ready..."
    $ready = $false
    foreach ($i in 1..90) {
        Start-Sleep -Seconds 2
        try {
            $null = Invoke-WebRequest 'http://127.0.0.1:8080/health' -TimeoutSec 5 -UseBasicParsing
            $ready = $true; break
        } catch { }
    }
    if (-not $ready) {
        Write-Host 'server did not become ready; see ~/.qwen35/llama-server.log' -ForegroundColor Red
        exit 1
    }
    Write-Host 'server ready' -ForegroundColor Green

    if (-not $Workspace) {
        $Workspace = Join-Path $root 'demo-workspace'
    }
    $ws = (Resolve-Path $Workspace).Path
    Write-Host "`n=== demo task in $ws ===" -ForegroundColor Cyan
    & $python -m qwen35 run --workspace $ws --yes --verbose `
        'Run the test suite with python test_calc.py and fix whatever is broken.'

    Write-Host "`nServer is still running (pid $($serve.Id)); stop it with:" -ForegroundColor Yellow
    Write-Host "  Stop-Process -Id $($serve.Id)" -ForegroundColor Yellow
}

Write-Host "`nDone. Try:" -ForegroundColor Green
Write-Host "  $python -m qwen35 chat --workspace <a project directory>"
