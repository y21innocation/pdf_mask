"""
AI-driven masking entry points (local mock provider by default).

This module provides a pluggable interface for AI-based masking.
Default provider is 'mock' that follows instructions in the prompt and
uses robust regex patterns to detect money expressions in Japanese text.

No network calls are made unless a non-mock provider is implemented later.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Set
import re
import unicodedata
import fitz  # PyMuPDF


# Patterns mirrored from app with a slightly more permissive variant
MONEY_PATTERN = re.compile(
    # Currency first OR digits+unit; allow inner spaces/commas/dots
    r"(?:[¥￥]\s*[0-9０-９](?:[\s,，\.．]*[0-9０-９])*)"
    r"|(?:[0-9０-９](?:[\s,，\.．]*[0-9０-９])*\s*(?:万\s*円|万円|円|¥|￥|万))",
    re.IGNORECASE,
)

# Enhanced pattern for corrupted text (e.g., cid encoding issues)
ENHANCED_MONEY_PATTERN = re.compile(
    # Original money patterns
    r"(?:[¥￥]\s*[0-9０-９](?:[\s,，\.．]*[0-9０-９])*)"
    r"|(?:[0-9０-９](?:[\s,，\.．]*[0-9０-９])*\s*(?:万\s*円|万円|円|¥|￥|万))"
    # Additional patterns for corrupted text
    r"|(?:[¥￥]\s*[0-9]+)"  # ¥ followed by numbers
    r"|(?:[0-9]+\s*[¥￥])"  # numbers followed by ¥
    r"|(?:[0-9]+\s*万)"     # numbers followed by 万
    r"|(?:[0-9]{3,}\s*円)"  # 3+ digits followed by 円
    r"|(?:年収\s*[0-9]+)"   # 年収 followed by numbers
    r"|(?:月給\s*[0-9]+)"   # 月給 followed by numbers
    r"|(?:[0-9]{2,}\s*[万円])", # 2+ digits with 万 or 円
    re.IGNORECASE,
)


def normalize_and_remove_invisible_chars(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    invisible_re = re.compile(r"[\u200B-\u200F\uFEFF\u202A-\u202E]")
    cleaned = invisible_re.sub("", normalized)
    return cleaned


def force_reconnect_money_string(line: str) -> str:
    # digits -> unit
    line = re.sub(r"([0-9０-９][0-9０-９,，\.]*)\s+((万|万円|円|¥|￥))", r"\1\2", line)
    # unit -> digits
    line = re.sub(r"((¥|￥))\s+([0-9０-９][0-9０-９,，\.]*)", r"\1\3", line)
    return line


try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai = None


class AiMasker:
    """A pluggable AI masker facade.

    provider: 'mock' only for now. Future: 'ollama', 'openai', etc.
    prompt: instruction string, supports keywords like 'money'.
    """

    def __init__(self, provider: str = "mock", prompt: Optional[str] = None, api_key: Optional[str] = None, model: str = "gemini-1.5-flash", mask_labels: Optional[Set[str]] = None):
        self.provider = (provider or "mock").lower()
        self.prompt = prompt or "Mask monetary amounts (金額, 年収, 月給, 円/万円/¥/￥)."
        self.api_key = api_key
        self.model = model
        # None or a set of labels to mask e.g., {"annual_income","monthly_income","bonus"}
        self.mask_labels = set(mask_labels) if mask_labels else None
        # feature flags from prompt (simple parsing)
        self.enable_money = True
        # extensible for phone/email later

        # quick toggle based on prompt content
        p = self.prompt
        self.enable_money = any(k in p for k in ["money", "金額", "年収", "円", "万円", "¥", "￥"]) or True

    def mask_page(self, lines: List[Dict[str, Any]], page_idx: int) -> Tuple[List[fitz.Rect], List[Tuple[int, str]]]:
        """
        Given grouped lines (as in app.group_words_into_lines output),
        return list of rects to redact and debug logs.
        """
        if self.provider == "gemini" and genai is not None and self.api_key:
            return self._mask_page_gemini(lines, page_idx)

        rects: List[fitz.Rect] = []
        logs: List[Tuple[int, str]] = []

        if self.enable_money:
            for ln in lines:
                # Reuse the same mapping approach as rule-based path
                tokens = ln.get("tokens", [])
                line_text = ln.get("text", "")
                
                # Exclusion check: Skip lines with company info keywords
                exclude_pattern = re.compile(
                    r"(資\s*本\s*金|売\s*上\s*高|売\s*上|資\s*産|負\s*債|純\s*資\s*産|時\s*価\s*総\s*額|企\s*業\s*価\s*値|株\s*式\s*数|株\s*価|取\s*引\s*額|契\s*約\s*金\s*額|投\s*資\s*額|融\s*資\s*額|借\s*入\s*金|預\s*金|残\s*高|口\s*座|従\s*業\s*員\s*数|設\s*立|創\s*業|上\s*場)",
                    re.IGNORECASE
                )
                if exclude_pattern.search(line_text):
                    continue  # Skip this line
                
                # Handle OCR text (when tokens are empty)
                if not tokens and line_text:
                    # For OCR text, just check the line text directly
                    patterns_to_try = [MONEY_PATTERN, ENHANCED_MONEY_PATTERN]
                    
                    for pattern in patterns_to_try:
                        for match in pattern.finditer(line_text):
                            matched_str = match.group(0)
                            label = self._classify_label(line_text)
                            if self.mask_labels is not None and label not in self.mask_labels:
                                continue
                            # Skip excluded labels
                            if label in ["excluded", "capital", "revenue"]:
                                continue
                            # Create a default rect for OCR text (will need improvement)
                            bbox = ln.get('bbox', [0, 0, 100, 20])
                            rects.append(fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3]))
                            logs.append((page_idx + 1, f"[AI OCR money] line=({line_text}) matched=({matched_str}) label=({label})"))
                            break  # Found a match, move to next line
                    continue
                
                # Handle normal token-based text
                if not tokens:
                    continue

                # Additional exclusion check for token-based text
                if exclude_pattern.search(line_text):
                    continue  # Skip this line

                combined_text_parts: List[str] = []
                index_map = []
                cur_offset = 0
                for i, tk in enumerate(tokens):
                    if i > 0:
                        combined_text_parts.append(" ")
                        cur_offset += 1
                    start_pos = cur_offset
                    combined_text_parts.append(tk[4])
                    cur_offset += len(tk[4])
                    end_pos = cur_offset
                    index_map.append((start_pos, end_pos, tk))

                combined_text = "".join(combined_text_parts)
                search_text = combined_text

                # Try both patterns: standard first, then enhanced for corrupted text
                patterns_to_try = [MONEY_PATTERN, ENHANCED_MONEY_PATTERN]
                
                for pattern in patterns_to_try:
                    for match in pattern.finditer(search_text):
                        m_start, m_end = match.start(), match.end()
                        matched_str = match.group(0)
                        # classify label from line text (simple keyword-based)
                        label = self._classify_label(ln.get('text',''))
                        if self.mask_labels is not None and label not in self.mask_labels:
                            continue
                        # Skip excluded labels
                        if label in ["excluded", "capital", "revenue"]:
                            continue
                        hit_tokens = []
                        for (tk_start, tk_end, token_item) in index_map:
                            if not (tk_end <= m_start or tk_start >= m_end):
                                hit_tokens.append(token_item)
                        if hit_tokens:
                            min_x = min(t[0] for t in hit_tokens)
                            min_y = min(t[1] for t in hit_tokens)
                            max_x = max(t[2] for t in hit_tokens)
                            max_y = max(t[3] for t in hit_tokens)
                            rects.append(fitz.Rect(min_x, min_y, max_x, max_y))
                            logs.append((page_idx + 1, f"[AI money] line=({ln.get('text','')}) matched=({matched_str}) label=({label})"))
                            break  # Found a match, move to next line

        return rects, logs

    def _classify_label(self, text: str) -> str:
        """Rudimentary classification based on keywords in the line text."""
        t = normalize_and_remove_invisible_chars(text)
        
        # Exclusion check: Skip company info
        exclude_pattern = re.compile(
            r"(資\s*本\s*金|売\s*上\s*高|売\s*上|資\s*産|負\s*債|純\s*資\s*産|時\s*価\s*総\s*額|企\s*業\s*価\s*値|株\s*式\s*数|株\s*価|取\s*引\s*額|契\s*約\s*金\s*額|投\s*資\s*額|融\s*資\s*額|借\s*入\s*金|預\s*金|残\s*高|口\s*座|従\s*業\s*員\s*数|設\s*立|創\s*業|上\s*場)",
            re.IGNORECASE
        )
        if exclude_pattern.search(t):
            return "excluded"  # This will be filtered out by mask_labels check
            
        # Capital/Revenue exclusions (additional specific checks)
        if re.search(r"資本\s*金|資本金", t):
            return "capital"
        if re.search(r"売上|決算|売上高", t):
            return "revenue"
        # Salary categories
        if re.search(r"(年\s*収|年\s*俸|希望\s*年\s*収|想定\s*年\s*収|年\s*収\s*目\s*安|直\s*近\s*の\s*年\s*収|現\s*年\s*収|前\s*年\s*収)", t):
            return "annual_income"
        if re.search(r"(月\s*収|月\s*給|月\s*額)", t):
            return "monthly_income"
        if re.search(r"(賞\s*与|年間\s*賞与|ボーナス)", t):
            return "bonus"
        if re.search(r"(手\s*当|残業手当|住宅手当|扶養手当|通勤手当|交通費)", t):
            return "allowance"
        return "other"

    # --- Gemini implementation (text-only, simple span extraction) ---
    def _mask_page_gemini(self, lines: List[Dict[str, Any]], page_idx: int) -> Tuple[List[fitz.Rect], List[Tuple[int, str]]]:
        print(f"[DEBUG GEMINI] _mask_page_gemini called with {len(lines)} lines")
        
        rects: List[fitz.Rect] = []
        logs: List[Tuple[int, str]] = []

        if not lines:
            print(f"[DEBUG GEMINI] No lines to process")
            return rects, logs

        # Prepare payload: list of lines with indices and text
        payload_lines = []
        for ln in lines:
            payload_lines.append({
                "idx": ln.get("line_index"),
                "text": ln.get("text", "")
            })

        # Ask Gemini to return labeled spans for selective masking
        instruction = (self.prompt or "") + "\n" + (
            "返答は必ずJSONのみで、{items:[{idx:number, start:number, end:number, label:string}]} の形。"
            "idxは行index、start/endは行テキスト中の文字範囲(半角基準)。"
            "labelは annual_income | monthly_income | bonus | allowance | other | capital などの区分とする。"
            "年収/年俸/想定年収/希望年収は annual_income、月収/月給/月額は monthly_income、賞与/年間賞与/ボーナスは bonus、"
            "手当は allowance、資本金は capital、それ以外の金額は other として分類。"
            "日本語の金額表記(円/万円/¥/￥/万/億)や全角半角混在、範囲表記(400〜600万円)にも対応して抽出。"
        )

        try:
            print(f"[DEBUG GEMINI] Configuring API with key: {self.api_key[:10]}...")
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            print(f"[DEBUG GEMINI] Sending {len(payload_lines)} lines to Gemini")
            content = [
                {"role": "user", "parts": [
                    instruction,
                    "\n\nLINES:",
                    str(payload_lines)
                ]}
            ]
            print(f"[DEBUG GEMINI] Making API call...")
            resp = model.generate_content(content)
            text = resp.text if hasattr(resp, "text") else ""
            print(f"[DEBUG GEMINI] Request: {payload_lines[:1]}...")  # Show first line
            print(f"[DEBUG GEMINI] Response length: {len(text)}")  # Show response length
            print(f"[DEBUG GEMINI] Response: {text}")  # Show full response
        except Exception as e:  # fallback to mock
            print(f"[DEBUG GEMINI] Exception occurred: {e}")
            print(f"[DEBUG GEMINI] Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            text = "{\"items\":[]}"

        # Very defensive JSON parse
        import json
        items = []
        try:
            # Clean response (remove markdown code blocks if present)
            clean_text = text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            data = json.loads(clean_text)
            items = data.get("items", []) if isinstance(data, dict) else []
            print(f"[DEBUG GEMINI] Parsed {len(items)} items from JSON")
        except Exception as e:
            print(f"[DEBUG GEMINI] JSON parse error: {e}")
            items = []

        # Map spans back to token rects by overlap
        # Build an index for quick lookup
        lines_by_idx = {ln["line_index"]: ln for ln in lines}
        print(f"[DEBUG GEMINI] Processing {len(items)} items")
        for it in items:
            try:
                idx = int(it.get("idx"))
                s = int(it.get("start"))
                e = int(it.get("end"))
                label = it.get("label") or "other"
                print(f"[DEBUG GEMINI] Processing item: idx={idx}, start={s}, end={e}, label={label}")
            except Exception as ex:
                print(f"[DEBUG GEMINI] Error parsing item {it}: {ex}")
                continue
            
            if self.mask_labels is not None and label not in self.mask_labels:
                print(f"[DEBUG GEMINI] Skipping label {label} (not in mask_labels)")
                continue
            ln = lines_by_idx.get(idx)
            if not ln:
                print(f"[DEBUG GEMINI] Line {idx} not found in lines_by_idx")
                continue
            if not ln:
                continue
            tokens = ln.get("tokens", [])
            # Build mapping like in mock
            combined_text_parts: List[str] = []
            index_map = []
            cur = 0
            for i, tk in enumerate(tokens):
                if i > 0:
                    combined_text_parts.append(" ")
                    cur += 1
                st = cur
                # Handle different token formats
                if isinstance(tk, dict):
                    token_text = tk.get("text", "")
                    bbox = [tk.get("x0", 0), tk.get("y0", 0), tk.get("x1", 0), tk.get("y1", 0)]
                elif isinstance(tk, (list, tuple)) and len(tk) >= 5:
                    token_text = tk[4]
                    bbox = [tk[0], tk[1], tk[2], tk[3]]
                else:
                    print(f"[DEBUG GEMINI] Unknown token format: {tk}")
                    continue
                
                combined_text_parts.append(token_text)
                cur += len(token_text)
                en = cur
                index_map.append((st, en, bbox))
            for (tk_st, tk_en, bbox) in index_map:
                if not (tk_en <= s or tk_st >= e):
                    rects.append(fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3]))
            logs.append((page_idx + 1, f"[AI(gemini)] line=({ln.get('text','')}) span=({s},{e}) label=({label})"))

        return rects, logs
