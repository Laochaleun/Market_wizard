# Session Notes: HF Spaces File Download Issue

## Problem Description
Reports (HTML and PDF) are generated successfully on Hugging Face Spaces but cannot be downloaded by users. The error message shown is "File wasn't available on site".

**Evidence:**
- Logs confirm files are created: `Report exported to: /tmp/ssr_report_20260130_193617.html | exists=True`
- Both HTML and PDF exports show `exists=True`
- But clicking download shows "File wasn't available on site"

## Environment
- **Platform**: Hugging Face Spaces (Docker-based)
- **Framework**: Gradio (version likely 5.x or 6.x)
- **Files saved to**: `/tmp/` directory
- **App file**: `frontend/main.py`

## Root Cause Analysis
Gradio has a known bug in version 6.x (GitHub Issue #12452) where dynamically generated files cannot be downloaded. The problem appears to be in how Gradio's file serving/caching mechanism handles files that are created after the app starts.

According to Gradio documentation, files from `tempfile.gettempdir()` should automatically be cached and served, but this doesn't work on HF Spaces.

## Failed Solutions

### Attempt 1: Save to project-relative `reports/` directory
- Changed output path from `/tmp/` to `PROJECT_BASE_DIR / "reports"`
- Added `allowed_paths=[str(reports_dir)]` to `demo.launch()`
- Modified Dockerfile to create `/app/reports` with `chmod 777`
- **Result**: Same error - files not available

### Attempt 2: Save directly to `tempfile.gettempdir()` without subdirectory
- Changed from `/tmp/market_wizard_reports/` to just `/tmp/`
- **Result**: Same error - files not available

### Attempt 3: Return `gr.File()` wrapper instead of string path
- Changed return from `str(output_path)` to `gr.File(value=str(output_path), visible=True)`
- **Result**: Caused app restart loop on HF Spaces

### Attempt 4: Add FastAPI download endpoint to bypass Gradio
- Added `/download-report/{filename}` endpoint using `FileResponse`
- Endpoint added via `demo.app` (Gradio's underlying FastAPI/Starlette app)
- Changed export functions to return download URL
- **Result**: `gr.File` component tried to stat the URL as filesystem path - crashed

### Attempt 5: Replace `gr.File` with `gr.HTML` for download link
- Changed output component from `gr.File` to `gr.HTML`
- Export functions return styled HTML `<a>` tag with download link
- Link points to `/download-report/{filename}` endpoint
- **Result**: Still shows "File wasn't available on site" - the HTML link appears but clicking it fails

## Current State of Code

### Export function returns:
```python
download_url = _build_download_url(output_path.name, request)
download_link = f'<a href="{download_url}" download="{output_path.name}" style="...">📥 Download {output_path.name}</a>'
return download_link, f"✅ Gotowe do pobrania"
```

### Download endpoint:
```python
app = FastAPI()

@app.get("/download-report/{filename}")
async def download_report(filename: str):
    filepath = Path(tempfile.gettempdir()) / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(path=str(filepath), filename=filename, media_type=media_type)

app = gr.mount_gradio_app(app, demo, path="/")
```

### UI component:
```python
export_file = gr.HTML(label="📥 Download / Pobierz")
```

## Hypotheses to Test in Next Session

1. **HF Spaces proxy issue**: HF Spaces might be proxying requests and blocking non-Gradio routes
2. **Route registration timing**: The download endpoint might not be properly registered before `demo.launch()` (resolved locally with FastAPI mount)
3. **Need to use `gr.mount_gradio_app()` pattern**: Instead of adding routes to Gradio's app, mount Gradio into a FastAPI app (implemented locally)
4. **Gradio's `root_path` configuration**: HF Spaces might use a different root path that breaks relative URLs
5. **Use absolute URL with HF Spaces domain**: Instead of `/download-report/...`, use full URL `https://pa-sk-market-wizard.hf.space/download-report/...`

## Resolution (Local)
- Root cause confirmed: `/download-report/...` route was not registered on the running app (404).
- Switched to FastAPI as the main app and mounted Gradio via `gr.mount_gradio_app(...)`.
- File downloads now served via FastAPI `FileResponse` and confirmed working for both HTML and PDF.
- Export links now built from request `base_url` / forwarded headers for proxy compatibility.
- Added targeted logging for URL construction, mount, and download serving.

## Files Changed During This Session
- `/Users/pawel/Market_wizard/frontend/main.py` - Multiple changes to export logic and UI components
- `/Users/pawel/Market_wizard/Dockerfile` - Added `/app/reports` directory creation
 - `/Users/pawel/Market_wizard/frontend/main.py` - Mounted Gradio into FastAPI and added request-aware download URLs/logs

## Relevant Documentation Links
- Gradio File Access: https://www.gradio.app/guides/file-access
- Gradio Issue #12452: https://github.com/gradio-app/gradio/issues/12452
- HF Spaces Docker: https://huggingface.co/docs/hub/spaces-sdks-docker
 - Gradio mount_gradio_app: https://www.gradio.app/docs/gradio/mount_gradio_app

## Screenshot of Error
![Download error](/Users/pawel/.gemini/antigravity/brain/55155640-d6a9-476c-8f95-31038b0f77a2/uploaded_media_1769801944368.png)
