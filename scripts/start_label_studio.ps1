# Launch Label Studio with local-file serving configured for this repo.
# Usage:  .\scripts\start_label_studio.ps1
#
# Reminder: serving also requires EACH project to register a Local Files
# source storage (Settings -> Cloud Storage -> Add Source Storage -> Local
# files, absolute path = the data folder below, do NOT click Sync).

$repo = Split-Path -Parent $PSScriptRoot

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = 'true'
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = Join-Path $repo 'data'

$ls = Join-Path $repo '.venv-ls\Scripts\label-studio.exe'
if (-not (Test-Path $ls)) {
    Write-Error "Label Studio venv not found. Run: py -3.11 -m venv .venv-ls; .\.venv-ls\Scripts\pip install label-studio"
    exit 1
}

Write-Host "Document root: $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
& $ls start
