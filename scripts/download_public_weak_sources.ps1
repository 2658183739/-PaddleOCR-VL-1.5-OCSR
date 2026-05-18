param(
    [string[]]$Sources = @("decimer", "patcid"),
    [string]$Root = "V2-1/data/public_sources",
    [switch]$IncludeLarge
)

$ErrorActionPreference = "Stop"

function Ensure-Dir {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

$rootPath = Resolve-Path "." | ForEach-Object { Join-Path $_ $Root }
Ensure-Dir -Path $rootPath

$sourceMap = @{
    "decimer" = @{
        Dir = "decimer_handdrawn"
        Url = "https://zenodo.org/records/7617107/files/DECIMER_Hand-drawn_Molecule_Images.zip?download=1"
        File = "decimer_handdrawn.zip"
        Large = $false
    }
    "patcid" = @{
        Dir = "patcid"
        Url = "https://zenodo.org/records/10572870/files/patcid.zip?download=1"
        File = "patcid.zip"
        Large = $true
    }
}

foreach ($source in $Sources) {
    if (-not $sourceMap.ContainsKey($source)) {
        Write-Warning "Unknown source: $source"
        continue
    }

    $item = $sourceMap[$source]
    if ($item.Large -and -not $IncludeLarge) {
        Write-Host "[SKIP] $source is large. Re-run with -IncludeLarge to download it." -ForegroundColor Yellow
        continue
    }

    $targetDir = Join-Path $rootPath $item.Dir
    Ensure-Dir -Path $targetDir
    Ensure-Dir -Path (Join-Path $targetDir "raw")
    Ensure-Dir -Path (Join-Path $targetDir "manifests")

    $archivePath = Join-Path $targetDir $item.File
    if (Test-Path $archivePath) {
        Write-Host "[EXISTS] $archivePath"
        continue
    }

    Write-Host "[DOWNLOAD] $source -> $archivePath"
    Invoke-WebRequest -Uri $item.Url -OutFile $archivePath
}

Write-Host ""
Write-Host "PubChem and ChEMBL are better handled as sampled SMILES sources, not one-shot bulk image downloads." -ForegroundColor Cyan
Write-Host "After downloading source tables manually, place them under:" -ForegroundColor Cyan
Write-Host "  V2-1/data/public_sources/pubchem/raw/" -ForegroundColor Cyan
Write-Host "  V2-1/data/public_sources/chembl/raw/" -ForegroundColor Cyan
Write-Host ""
Write-Host "Then use sample_public_smiles_seed.py to build a seed CSV for generated evaluation." -ForegroundColor Cyan
