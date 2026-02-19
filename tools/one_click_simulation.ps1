# UWB Channel Modeling - One-click simulation/validation runner (PowerShell)
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

Write-Host "[0/9] Repo 확인" -ForegroundColor Cyan
$repo = (Get-Location).Path
Write-Host "Repo: $repo"

function Get-PythonCandidates {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += ,@("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += ,@("python")
    }
    if ($candidates.Count -eq 0) {
        throw "Python 실행기(python 또는 py)를 찾을 수 없습니다."
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
    foreach ($candidate in (Get-PythonCandidates)) {
        try {
            $code = Invoke-Python -Launcher $candidate -CmdArgs @("-c", "import sys; print(sys.executable)")
            if ($code -eq 0) {
                return $candidate
            }
        } catch {
            continue
        }
    }
    throw "작동 가능한 Python 실행기를 찾지 못했습니다."
}

$launcher = Select-WorkingPython
Write-Host ("Using launcher: " + ($launcher -join " ")) -ForegroundColor DarkCyan

Write-Host "[1/9] venv 준비" -ForegroundColor Yellow
$venvDir = Join-Path $repo ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"

if (!(Test-Path $venvPy)) {
    if (Test-Path $venvDir) {
        Write-Host " - 기존 .venv가 불완전하여 재생성합니다." -ForegroundColor DarkYellow
        Remove-Item -Recurse -Force $venvDir
    }

    Write-Host " - python -m venv .venv" -ForegroundColor DarkYellow
    $createCode = Invoke-Python -Launcher $launcher -CmdArgs @("-m", "venv", ".venv")
    if ($createCode -ne 0) {
        throw "venv 생성 실패 (exit=$createCode)"
    }
}

if (!(Test-Path $venvPy)) {
    throw "venv python을 찾을 수 없습니다: $venvPy"
}

Write-Host "[2/9] 패키지 도구 업그레이드" -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip setuptools wheel

Write-Host "[3/9] 의존성 설치" -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    & $venvPy -m pip install -r requirements.txt
} else {
    & $venvPy -m pip install numpy matplotlib scipy h5py pytest
}

Write-Host "[4/9] baseline validation" -ForegroundColor Yellow
& $venvPy -m scripts.run_validation --out artifacts/baseline_report.md

if (-not $SkipTests) {
    Write-Host "[5/9] pytest" -ForegroundColor Yellow
    & $venvPy -m pytest -q
} else {
    Write-Host "[5/9] pytest 생략(-SkipTests)" -ForegroundColor DarkYellow
}

Write-Host "[6/9] 시뮬레이션/검증 실행 (updated report)" -ForegroundColor Yellow
& $venvPy -m scripts.run_validation --out $OutReport

Write-Host "[7/9] HDF5 출력 확인" -ForegroundColor Yellow
if (Test-Path $OutH5) {
    Write-Host " - H5 exists: $OutH5" -ForegroundColor Green
} else {
    Write-Host " - H5 경로가 비어있습니다. runner 기본값은 artifacts/rt_sweep.h5 입니다." -ForegroundColor DarkYellow
}

Write-Host "[8/9] 결과 요약" -ForegroundColor Green
Write-Host " - baseline: artifacts/baseline_report.md"
Write-Host " - updated : $OutReport"
Write-Host " - plots   : artifacts/plots"

Write-Host "[9/9] updated report preview" -ForegroundColor Cyan
if (Test-Path $OutReport) {
    Get-Content $OutReport -TotalCount 60
} else {
    Write-Host "updated report를 찾지 못했습니다: $OutReport" -ForegroundColor Red
}
