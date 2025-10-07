#!/usr/bin/env python3
"""
Direct AI masking test script
"""
import os
import sys

# Set environment variables
os.environ['AI_STRATEGY'] = 'ai'
os.environ['AI_PROVIDER'] = 'gemini'
os.environ['GEMINI_API_KEY'] = 'AIzaSyAsTcYWS4d-db6xjX6E9ibrCpa97xwXEFM'
os.environ['AI_PROMPT'] = 'Mask monetary amounts (金額, 年収, 月給, 円/万円/¥/￥).'

# Import after setting environment variables
from ai_mask import AiMasker
import fitz  # PyMuPDF

def group_words_into_lines(page):
    """Extract and group words into lines from PDF page"""
    blocks = page.get_text("dict")["blocks"]
    words = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    word_text = span["text"].strip()
                    if word_text:
                        bbox = span["bbox"]
                        words.append({
                            "text": word_text,
                            "bbox": bbox,
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3]
                        })
    
    # Group words into lines
    lines = []
    current_line = []
    current_y = None
    
    # Sort words by y position then x position
    words.sort(key=lambda w: (w["y0"], w["x0"]))
    
    for word in words:
        if current_y is None or abs(word["y0"] - current_y) < 5:  # Same line
            current_line.append(word)
            current_y = word["y0"]
        else:  # New line
            if current_line:
                line_text = " ".join([w["text"] for w in current_line])
                line_bbox = [
                    min(w["x0"] for w in current_line),
                    min(w["y0"] for w in current_line),
                    max(w["x1"] for w in current_line),
                    max(w["y1"] for w in current_line)
                ]
                lines.append({
                    "text": line_text,
                    "tokens": current_line,
                    "bbox": line_bbox,
                    "line_index": len(lines)
                })
            current_line = [word]
            current_y = word["y0"]
    
    # Add the last line
    if current_line:
        line_text = " ".join([w["text"] for w in current_line])
        line_bbox = [
            min(w["x0"] for w in current_line),
            min(w["y0"] for w in current_line),
            max(w["x1"] for w in current_line),
            max(w["y1"] for w in current_line)
        ]
        lines.append({
            "text": line_text,
            "tokens": current_line,
            "bbox": line_bbox,
            "line_index": len(lines)
        })
    
    return lines

def test_ai_masking():
    """Test AI masking with Gemini API"""
    pdf_path = "/Users/kounoyousuke/pdf_mask/image_based_test.pdf"
    
    print(f"Testing AI masking with: {pdf_path}")
    print(f"AI_STRATEGY: {os.environ.get('AI_STRATEGY')}")
    print(f"AI_PROVIDER: {os.environ.get('AI_PROVIDER')}")
    print(f"AI_PROMPT: {os.environ.get('AI_PROMPT')}")
    print()
    
    # Open PDF
    doc = fitz.open(pdf_path)
    print(f"PDF has {len(doc)} pages")
    
    # Initialize AI masker
    ai = AiMasker(
        provider=os.environ.get('AI_PROVIDER'),
        prompt=os.environ.get('AI_PROMPT'),
        api_key=os.environ.get('GEMINI_API_KEY'),
        model="gemini-2.5-flash"  # Use the available stable model
    )
    print(f"Created AiMasker: {ai}")
    
    # Check Gemini availability
    import ai_mask
    print(f"genai import status: {ai_mask.genai is not None}")
    print(f"API key: {os.environ.get('GEMINI_API_KEY')[:10]}...")
    print(f"Provider: {ai.provider}")
    print()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"--- Page {page_num + 1} ---")
        
        # Extract lines
        lines = group_words_into_lines(page)
        print(f"Extracted {len(lines)} lines")
        
        for i, line in enumerate(lines):  # Show all lines
            print(f"  Line {i+1}: {repr(line['text'])}")
        print()
        
        # Run AI masking
        try:
            print(f"Calling AI masking with provider: {ai.provider}")
            rects, ai_logs = ai.mask_page(lines, page_num)
            print(f"AI masking result: {len(rects)} rectangles, {len(ai_logs)} log entries")
            
            if rects:
                print("Found mask rectangles:")
                for i, rect in enumerate(rects):
                    print(f"  Rect {i+1}: {rect}")
            
            if ai_logs:
                print("AI logs:")
                for log in ai_logs:
                    print(f"  {log}")
            
        except Exception as e:
            print(f"AI masking error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    doc.close()
    print("Test completed")

if __name__ == "__main__":
    test_ai_masking()