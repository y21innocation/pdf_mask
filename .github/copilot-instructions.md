# PDF Salary Masking Tool - AI Copilot Instructions

This is a specialized PDF processing web application designed to automatically detect and mask salary-related information in Japanese PDF documents. The tool is particularly useful for HR departments and recruitment agencies handling sensitive employee information.

## Architecture Overview

### Core Technology Stack
- **Flask**: Web framework for file upload/download interface
- **PyMuPDF (fitz)**: Primary PDF manipulation and text extraction
- **pdfplumber**: Secondary PDF processing for table detection
- **Deployment**: Render.com with Gunicorn WSGI server

### Key Components
- `app.py`: Main application with salary detection and masking logic
- `templates/index.html`: Simple file upload interface
- `output/`: Generated masked PDFs and processing logs
- Docker containerization with Japanese font support

## Critical Processing Logic

### Salary Detection Algorithm
The application uses a two-tier detection system:

1. **Keyword Pattern Matching** (`KEYWORDS_PATTERN`):
   - Detects salary-related terms: 年収, 給与, 月給, 賞与, etc.
   - Located at lines 22-25 in `app.py`

2. **Money Amount Pattern** (`MONEY_PATTERN`):
   - Regex: `[0-9０-９][0-9０-９,\.]*\s*(万\s*円|万円|円|¥|￥|万)`
   - Handles both half-width and full-width Japanese numerals
   - Located at lines 17-20 in `app.py`

### Masking Strategy
- **Keyword lines**: Mask ALL numeric tokens + money patterns
- **Adjacent lines** (±2 lines): Mask only money patterns
- **Other lines**: No masking applied

### Text Processing Challenges
The codebase handles several Japanese PDF text extraction issues:

1. **Unicode Normalization**: `normalize_and_remove_invisible_chars()` (lines 28-35)
2. **Token Reconnection**: `force_reconnect_money_string()` handles space-separated amounts (lines 37-43)
3. **Adjacent Token Unification**: Special logic for number-unit pairs with extended gap tolerance (lines 58-78)

## Development Workflows

### Local Development
```bash
pip install -r requirements.txt
python app.py  # Runs on localhost:5000
```

### Container Build
```bash
docker build -t pdf-mask .
docker run -p 5000:5000 pdf-mask
```

### Render.com Deployment
- Uses `render-build.sh` to install system dependencies (poppler-utils, fonts)
- Gunicorn configuration: 2 workers, port 5000
- Automatic deployment via `render.yaml`

### Masking Modes (Env Vars)
- `MASK_STRATEGY`:
   - `near` (default): Mask amounts only on salary keyword lines and ±NEAR_RANGE lines
   - `all`: Mask ALL money patterns on the page regardless of keywords
- `FALLBACK_MASK_PAGE_IF_NONE` (default `1`): In `near` mode, if a page gets 0 masks, fallback to masking all money patterns on that page
   - Set to `0` to disable fallback

## Project-Specific Patterns

### File Naming Convention
- Input: `original_filename.pdf`
- Masked PDF: `maskedfix_original_filename.pdf`
- Log file: `masked_original_filename.txt`
- Batch output: `maskedfix_and_texts.zip`

### Debug Logging
The application includes extensive debug output for development:
- Table row extraction debugging (lines 294-301)
- Non-table text line debugging (lines 303-310)
- Masking operation logging with Japanese text handling

### Error Handling Considerations
- **Temporary Directory Management**: Uses `tempfile.TemporaryDirectory()` for safe cleanup
- **PDF Processing**: Dual library approach (PyMuPDF + pdfplumber) for robustness
- **Japanese Text**: Special handling for NFKC normalization and invisible characters

## Key Integration Points

### PDF Libraries Coordination
- **PyMuPDF**: Primary for text extraction and redaction annotations
- **pdfplumber**: Secondary for table detection and text extraction
- Coordinate between both libraries for comprehensive text analysis

### Font Support
Dockerfile includes Japanese font packages:
- `fonts-ipafont-gothic`
- `fonts-ipafont-mincho` 
- `fonts-noto-cjk`

Required for proper Japanese text rendering in container environments.

## Development Notes

When modifying the masking logic:
1. Test with both table and non-table content
2. Verify Unicode normalization with full-width Japanese numbers
3. Check token reconnection for space-separated amounts
4. Validate adjacent line detection with `NEAR_RANGE` parameter (currently 2)

The `group_words_into_lines()` function (lines 81-143) is critical for proper line-based masking and should be modified carefully as it handles coordinate-based text grouping.

## AI Masking (Optional)

- Toggle with env vars:
   - `AI_STRATEGY`: `none` (default) | `ai` (AI only) | `ai+rules` (AI then rules)
   - `AI_PROVIDER`: `mock` (default; local, no network)
   - `AI_PROMPT`: Natural language instruction, e.g., "Mask monetary amounts (金額, 年収, 月給, 円/万円/¥/￥)."
- Implementation lives in `ai_mask.py` and integrates in `upload()` loop.
- Mock provider respects the prompt keywords and uses robust money regex to produce redaction rects.

### Gemini Provider (Online)
- Requirements:
   - Package: `google-generativeai`
   - Env vars:
      - `AI_STRATEGY=ai` or `ai+rules`
      - `AI_PROVIDER=gemini`
      - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
      - Optional: `AI_MODEL` (default `gemini-1.5-flash`)
- The app will send line texts per page to Gemini with your `AI_PROMPT` and expects JSON spans to map back to redaction rects.
- Network access and API quota are required; fallback to mock if misconfigured.