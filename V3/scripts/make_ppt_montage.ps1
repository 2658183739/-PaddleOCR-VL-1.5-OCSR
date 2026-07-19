param(
    [Parameter(Mandatory = $true)]
    [string]$ImageDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [int]$Columns = 3
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$images = Get-ChildItem -LiteralPath $ImageDirectory -Filter "*.PNG" |
    Sort-Object { [int]([regex]::Match($_.BaseName, "\d+").Value) }
if ($images.Count -eq 0) {
    throw "No PNG slides found in $ImageDirectory"
}

$thumbWidth = 426
$thumbHeight = 240
$gap = 8
$rows = [int][Math]::Ceiling($images.Count / [double]$Columns)
$canvasWidth = $Columns * $thumbWidth + ($Columns - 1) * $gap
$canvasHeight = $rows * $thumbHeight + ($rows - 1) * $gap
$canvas = New-Object System.Drawing.Bitmap($canvasWidth, $canvasHeight)
$graphics = [System.Drawing.Graphics]::FromImage($canvas)
$graphics.Clear([System.Drawing.Color]::FromArgb(232, 237, 239))
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic

try {
    for ($index = 0; $index -lt $images.Count; $index++) {
        $source = [System.Drawing.Image]::FromFile($images[$index].FullName)
        try {
            $column = $index % $Columns
            $row = [int][Math]::Floor($index / $Columns)
            $x = $column * ($thumbWidth + $gap)
            $y = $row * ($thumbHeight + $gap)
            $graphics.DrawImage($source, $x, $y, $thumbWidth, $thumbHeight)
        }
        finally {
            $source.Dispose()
        }
    }

    $output = [System.IO.Path]::GetFullPath($OutputPath)
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($output)) | Out-Null
    $canvas.Save($output, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $graphics.Dispose()
    $canvas.Dispose()
}

Get-Item -LiteralPath $OutputPath | Select-Object FullName, Length
