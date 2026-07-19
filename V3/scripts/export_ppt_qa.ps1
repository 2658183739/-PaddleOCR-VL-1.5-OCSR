param(
    [Parameter(Mandatory = $true)]
    [string]$PresentationPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$presentation = (Resolve-Path -LiteralPath $PresentationPath).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($output) | Out-Null

$powerPoint = New-Object -ComObject PowerPoint.Application
try {
    $deck = $powerPoint.Presentations.Open($presentation, $true, $false, $false)
    try {
        $slideWidth = [double]$deck.PageSetup.SlideWidth
        $slideHeight = [double]$deck.PageSetup.SlideHeight
        $epsilon = 0.75
        $shapeCount = 0
        $outOfBounds = @()
        foreach ($slide in $deck.Slides) {
            foreach ($shape in $slide.Shapes) {
                $shapeCount++
                $left = [double]$shape.Left
                $top = [double]$shape.Top
                $right = $left + [double]$shape.Width
                $bottom = $top + [double]$shape.Height
                if (
                    $left -lt -$epsilon -or $top -lt -$epsilon -or
                    $right -gt ($slideWidth + $epsilon) -or
                    $bottom -gt ($slideHeight + $epsilon)
                ) {
                    $outOfBounds += [pscustomobject]@{
                        slide = [int]$slide.SlideIndex
                        shape = [string]$shape.Name
                        left = [math]::Round($left, 2)
                        top = [math]::Round($top, 2)
                        right = [math]::Round($right, 2)
                        bottom = [math]::Round($bottom, 2)
                    }
                }
            }
        }
        [pscustomobject]@{
            schema = "paddleocr-vl-ocsr/ppt-shape-bounds-audit/v1"
            presentation = $presentation
            slides = [int]$deck.Slides.Count
            shapes = $shapeCount
            slide_width_points = $slideWidth
            slide_height_points = $slideHeight
            out_of_bounds_count = $outOfBounds.Count
            out_of_bounds = $outOfBounds
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output "shape_bounds_audit.json") -Encoding UTF8

        $deck.Export($output, "PNG", 1280, 720)
    }
    finally {
        $deck.Close()
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck) | Out-Null
    }
}
finally {
    $powerPoint.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) | Out-Null
}

Get-ChildItem -LiteralPath $output -Filter "*.PNG" |
    Sort-Object Name |
    Select-Object Name, Length
