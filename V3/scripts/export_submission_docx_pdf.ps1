$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$finalDir = Join-Path $workspace "决赛"
$documents = @(
    "数据构建报告_V3_final.docx",
    "评测集数据构建报告_V3_final.docx"
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($name in $documents) {
        $inputPath = Join-Path $finalDir $name
        $outputPath = [System.IO.Path]::ChangeExtension($inputPath, ".pdf")
        $document = $word.Documents.Open($inputPath, $false, $true)
        try {
            $document.Fields.Update() | Out-Null
            $document.ExportAsFixedFormat($outputPath, 17)
        }
        finally {
            $document.Close($false)
        }
        Write-Output $outputPath
    }
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
}
