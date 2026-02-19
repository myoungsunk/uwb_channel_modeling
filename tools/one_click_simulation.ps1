# UWB Channel Modeling - one-click simulation/validation runner (PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\tools\one_click_simulation.ps1
# Optional:
#   -OutReport artifacts/updated_report.md
#   -OutH5 artifacts/rt_sweep.h5
#   -SkipTests

param(
    [string]$OutReport = "artifacts/updated_report.md",
    [string]$OutH5 = "artifacts/rt_sweep.h5",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

Write-Host "[0/9] Repo check" -ForegroundColor Cyan
$repo = (Get-Location).Path
Write-Host "Repo: $repo"

function Get-PythonCandidates {
    $candidates = New-Object System.Collections.ArrayList

    # 1) Launcher commands from PATH
    if (Get-Command py -ErrorAction SilentlyContinue) {
        [void]$candidates.Add(@("py", "-3"))
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        [void]$candidates.Add(@("python"))
    }

    # 2) Well-known absolute locations on Windows
    $winPy = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "$env:ProgramFiles\Python310\python.exe",
        "$env:ProgramFiles(x86)\Python312\python.exe",
        "$env:ProgramFiles(x86)\Python311\python.exe",
        "$env:ProgramFiles(x86)\Python310\python.exe"
    )
    foreach ($path in $winPy) {
        if ($path -and (Test-Path $path)) {
            [void]$candidates.Add(@($path))
        }
    }

    # 3) Last fallback: Python launcher absolute path
    $pyLauncher = "$env:SystemRoot\py.exe"
    if (Test-Path $pyLauncher) {
        [void]$candidates.Add(@($pyLauncher, "-3"))
    }

    return $candidates
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Launcher,
        [Parameter(Mandatory = $true)]
        [string[]]$CmdArgs
    )

    if ($Launcher.Count -eq 1) {
        & $Launcher[0] @CmdArgs
    } else {
        & $Launcher[0] $Launcher[1] @CmdArgs
    }
    return $LASTEXITCODE
}

function Select-WorkingPython {
    $candidates = Get-PythonCandidates
    foreach ($candidate in $candidates) {
        try {
            $code = Invoke-Python -Launcher $candidate -CmdArgs @("-c", "import sys; print(sys.executable)")
            if ($code -eq 0) {
                return $candidate
            }
        } catch {
            continue
        }
    }

    $msg = @(
        "No usable Python launcher found.",
        "Tried PATH commands (py/python), common install paths, and %SystemRoot%\\py.exe.",
        "Install Python 3.10+ and ensure either 'py -3' or 'python' works in PowerShell."
    ) -join " "
    throw $msg
}

$launcher = Select-WorkingPython
Write-Host ("Using launcher: " + ($launcher -join " ")) -ForegroundColor DarkCyan

Write-Host "[1/9] Setup venv" -ForegroundColor Yellow
$venvDir = Join-Path $repo ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"

if (!(Test-Path $venvPy)) {
    if (Test-Path $venvDir) {
        Write-Host " - Existing .venv looks incomplete. Recreating..." -ForegroundColor DarkYellow
        Remove-Item -Recurse -Force $venvDir
    }

    Write-Host " - Creating .venv via: -m venv .venv" -ForegroundColor DarkYellow
    $createCode = Invoke-Python -Launcher $launcher -CmdArgs @("-m", "venv", ".venv")
    if ($createCode -ne 0) {
        throw "venv creation failed (exit=$createCode)"
    }
}

if (!(Test-Path $venvPy)) {
    throw "venv python not found after creation: $venvPy"
}

Write-Host "[2/9] Upgrade packaging tools" -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip setuptools wheel

Write-Host "[3/9] Install dependencies" -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    & $venvPy -m pip install -r requirements.txt
} else {
    & $venvPy -m pip install numpy matplotlib scipy h5py pytest
}

Write-Host "[4/9] Baseline validation" -ForegroundColor Yellow
& $venvPy -m scripts.run_validation --out artifacts/baseline_report.md

if (-not $SkipTests) {
    Write-Host "[5/9] Run pytest" -ForegroundColor Yellow
    & $venvPy -m pytest -q
} else {
    Write-Host "[5/9] Skip pytest (-SkipTests)" -ForegroundColor DarkYellow
}

Write-Host "[6/9] Updated validation" -ForegroundColor Yellow
& $venvPy -m scripts.run_validation --out $OutReport

Write-Host "[7/9] Check HDF5 output" -ForegroundColor Yellow
if (Test-Path $OutH5) {
    Write-Host " - H5 exists: $OutH5" -ForegroundColor Green
} else {
    Write-Host " - H5 not found at $OutH5 (runner default is artifacts/rt_sweep.h5)." -ForegroundColor DarkYellow
}

Write-Host "[8/9] Summary" -ForegroundColor Green
Write-Host " - baseline: artifacts/baseline_report.md"
Write-Host " - updated : $OutReport"
Write-Host " - plots   : artifacts/plots"

Write-Host "[9/9] Updated report preview" -ForegroundColor Cyan
if (Test-Path $OutReport) {
    Get-Content $OutReport -TotalCount 60
} else {
    Write-Host "updated report not found: $OutReport" -ForegroundColor Red
}
