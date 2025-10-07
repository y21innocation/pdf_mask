#!/usr/bin/env python3
"""
List available Gemini models
"""
import os
import google.generativeai as genai

# Set API key
genai.configure(api_key='AIzaSyAsTcYWS4d-db6xjX6E9ibrCpa97xwXEFM')

print("Available Gemini models:")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
            print(f"    Display name: {model.display_name}")
            print(f"    Description: {model.description}")
            print()
except Exception as e:
    print(f"Error listing models: {e}")