#!/usr/bin/env python3
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Warning: kagglehub not installed. Using local model only.")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from typing import List, Dict
import argparse
import os

class NERPredictor:
    def __init__(self, model_path: str = "./ner_model"):
        """Initialize with fallback to local model"""
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
        
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="average",
            device=self.device
        )

def download_model():
    if not KAGGLEHUB_AVAILABLE:
        raise ImportError("kagglehub package not available")
    try:
        print("Downloading model from Kaggle Hub...")
        path = kagglehub.model_download("afinaapayeva/ner_tuned_az/pyTorch/default")
        print(f"Model downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Download failed: {str(e)}")
        return "./ner_model"  # Fallback to local

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--local", action="store_true", help="Force use local model")
    args = parser.parse_args()

    # Model loading logic
    model_path = "./ner_model"
    if not args.local and KAGGLEHUB_AVAILABLE:
        try:
            model_path = download_model()
        except Exception as e:
            print(f"Using local model due to: {str(e)}")

    # Initialize and run
    try:
        predictor = NERPredictor(model_path)
        if args.text:
            from termcolor import colored  # Local import for better error handling
            predictor.pretty_print(args.text)
        else:
            print("Please provide text with --text")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Suggested fixes:")
        print("1. Ensure you have a local model at ./ner_model")
        print("2. Install kagglehub: pip install kagglehub")
        print("3. Check your internet connection if downloading")