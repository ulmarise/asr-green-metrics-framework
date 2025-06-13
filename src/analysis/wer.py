#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import jiwer
import sys
import argparse
import json
import re
import unicodedata
from typing import Iterator, List, Match, Optional, Union
from fractions import Fraction


# Non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
    "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
}

def remove_symbols_and_diacritics(s: str, keep=""):
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )

def remove_symbols(s: str):
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )

class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            try:
                import regex
                s = " ".join(regex.findall(r"\X", s, regex.U))
            except ImportError:
                pass

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s

class EnglishNumberNormalizer:
    def __init__(self):
        self.zeros = {"o", "oh", "zero"}
        self.ones = {
            name: i
            for i, name in enumerate(
                [
                    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                    "sixteen", "seventeen", "eighteen", "nineteen",
                ],
                start=1,
            )
        }
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s")
            for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }
        self.tens_plural = {
            name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()
        }
        self.tens_ordinal = {
            name.replace("y", "ieth"): (value, "th")
            for name, value in self.tens.items()
        }
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
        }
        self.multipliers_plural = {
            name + "s": (value, "s") for name, value in self.multipliers.items()
        }
        self.multipliers_ordinal = {
            name + "th": (value, "th") for name, value in self.multipliers.items()
        }
        self.multipliers_suffixed = {
            **self.multipliers_plural,
            **self.multipliers_ordinal,
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-", "negative": "-", "plus": "+", "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£", "pounds": "£", "euro": "€", "euros": "€",
            "dollar": "$", "dollars": "$", "cent": "¢", "cents": "¢",
        }
        self.prefixes = set(
            list(self.preceding_prefixers.values())
            + list(self.following_prefixers.values())
        )
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = set(
            [
                key
                for mapping in [
                    self.zeros, self.ones, self.ones_suffixed,
                    self.tens, self.tens_suffixed,
                    self.multipliers, self.multipliers_suffixed,
                    self.preceding_prefixers, self.following_prefixers,
                    self.suffixers, self.specials,
                ]
                for key in mapping
            ]
        )
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix = None
        value = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        def windowed_list(iterable, n=3):
            result = []
            for i in range(len(iterable) - n + 1):
                result.append(tuple(iterable[i:i+n]))
            return result
        
        windows = windowed_list([None] + words + [None], 3)
        for prev, current, next in windows:
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current and len(current) > 0 and current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            
            if current is None:
                continue
                
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:  # Safety check
                    continue
                    
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:  # replace the last zero with the digit
                        if str(value).endswith("0"):
                            value = value[:-1] + str(ones)
                        else:
                            value = str(value) + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed or current in self.tens_suffixed or current in self.multipliers_suffixed:
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.tens or current in self.multipliers:
                if value is not None:
                    yield output(value)
                yield output(current)
            else:
                if value is not None:
                    yield output(value)
                yield output(current)

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        # but remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)
        return s

    def postprocess(self, s: str):
        # write "one(s)" instead of "1(s)", just for the readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)
        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)
        return s

class EnglishSpellingNormalizer:
    def __init__(self):
        self.mapping = {
            # a subset of common spelling differences
            "colour": "color", "flavour": "flavor", "humour": "humor", 
            "labour": "labor", "neighbour": "neighbor",
            "centre": "center", "theatre": "theater", "metre": "meter",
            "analyse": "analyze", "organise": "organize", "recognise": "recognize",
            "defence": "defense", "offence": "offense", "pretence": "pretense",
            "catalogue": "catalog", "dialogue": "dialog", "programme": "program",
            "travelling": "traveling", "cancelled": "canceled", "jewellery": "jewelry",
        }

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())

class EnglishTextNormalizer:
    def __init__(self):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not", r"\bcan't\b": "can not", r"\blet's\b": "let us",
            r"\bain't\b": "aint", r"\by'all\b": "you all", r"\bwanna\b": "want to",
            r"\bgotta\b": "got to", r"\bgonna\b": "going to", 
            r"\bi'ma\b": "i am going to", r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have", r"\bcoulda\b": "could have", 
            r"\bshoulda\b": "should have", r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ", r"\bmrs\b": "missus ", r"\bst\b": "saint ",
            r"\bdr\b": "doctor ", r"\bprof\b": "professor ", r"\bcapt\b": "captain ",
            # general contractions
            r"n't\b": " not", r"'re\b": " are", r"'s\b": " is", r"'d\b": " would",
            r"'ll\b": " will", r"'t\b": " not", r"'ve\b": " have", r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep numeric symbols

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        return s

def load_references_from_text_file(text_file_path):
    references = {}
    try:
        with open(text_file_path, "r", encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    ref_id = parts[0]
                    text = parts[1]
                    
                    references[ref_id] = text
                    
                    match = re.search(r'(\d+-\d+-\d+)', ref_id)
                    if match:
                        alt_id = match.group(1)
                        references[alt_id] = text
                        
        return references
    except Exception as e:
        print(f"Error loading text file: {e}")
        return {}

def load_references_from_csv(csv_file_path, filename_col='filename', text_col='text'):
    references = {}
    try:
        df = pd.read_csv(csv_file_path)
        
        if filename_col not in df.columns or text_col not in df.columns:
            print(f"Error: CSV must contain '{filename_col}' and '{text_col}' columns")
            return {}
        
        for _, row in df.iterrows():
            filename = row[filename_col]
            text = row[text_col]
            
            base_name = os.path.basename(filename)
            base_name_no_ext = os.path.splitext(base_name)[0]
            
            references[filename] = text
            references[base_name] = text  
            references[base_name_no_ext] = text
            
            numeric_match = re.search(r'(\d+)', base_name_no_ext)
            if numeric_match:
                numeric_id = numeric_match.group(1)
                references[f"sample-{numeric_id}"] = text
                references[f"sample-{numeric_id:06d}"] = text
                
        return references
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}

def auto_detect_reference_format(file_path):
    if file_path.lower().endswith('.csv'):
        return 'csv'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if ',' in first_line and ('filename' in first_line.lower() or 'text' in first_line.lower()):
                return 'csv'
            else:
                return 'text'
    except:
        return 'text'  

def load_hypotheses(hypothesis_dir, verbose=False):
    hypotheses = {}
    
    for root, dirs, files in os.walk(hypothesis_dir):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding='utf-8') as file:
                        file_content = file.read().strip()
                        if file_content:  
                            base_name = os.path.splitext(filename)[0]
                            hypotheses[base_name] = file_content
                            if verbose:
                                print(f"Loaded hypothesis: {base_name} -> {file_content[:50]}...")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return hypotheses

def match_hypotheses_with_references(hypotheses, references, verbose=False):
    matched_pairs = []
    
    for hyp_key, hyp_text in hypotheses.items():
        if hyp_key in references:
            matched_pairs.append((hyp_text, references[hyp_key]))
            if verbose:
                print(f"Direct match: {hyp_key}")
    
    if len(matched_pairs) < len(hypotheses) * 0.5:
        if verbose:
            print("Few direct matches found, trying flexible matching...")
        matched_pairs = []
        
        hyp_keys = sorted(list(hypotheses.keys()))
        ref_keys = sorted(list(references.keys()))
        
        if len(hyp_keys) == len(ref_keys):
            for i in range(len(hyp_keys)):
                matched_pairs.append((hypotheses[hyp_keys[i]], references[ref_keys[i]]))
                if verbose:
                    print(f"Position-matched: {hyp_keys[i]} with {ref_keys[i]}")
        else:
            for hyp_key, hyp_text in hypotheses.items():
                possible_keys = [
                    hyp_key,
                    f"sample-{hyp_key}",
                    f"cv-other-test/sample-{hyp_key}.mp3",
                    f"sample-{hyp_key}.mp3"
                ]
                
                hyp_num_matches = re.findall(r'(\d+)', hyp_key)
                for hyp_num in hyp_num_matches:
                    possible_keys.extend([
                        hyp_num,
                        f"{hyp_num}-{hyp_num}-{hyp_num}",  # Pattern like "123-123-123"
                        f"sample-{hyp_num}",
                        f"sample-{hyp_num:06d}",
                        f"cv-other-test/sample-{hyp_num:06d}.mp3",
                        f"sample-{hyp_num:06d}.mp3"
                    ])
                
                matched = False
                for possible_key in possible_keys:
                    if possible_key in references:
                        matched_pairs.append((hyp_text, references[possible_key]))
                        if verbose:
                            print(f"Pattern match: {hyp_key} -> {possible_key}")
                        matched = True
                        break
                
                if not matched and verbose:
                    print(f"No match found for: {hyp_key}")
    
    return matched_pairs

def main():
    parser = argparse.ArgumentParser(description="Calculate WER for ASR output with flexible reference format support")
    parser.add_argument("hypothesis_dir", nargs="?", 
                        default="./output",
                        help="Directory containing hypothesis (transcription) files")
    parser.add_argument("--reference_file", 
                        default="./data/trans.txt",
                        help="Reference file (text format or CSV)")
    parser.add_argument("--reference_format", choices=['auto', 'text', 'csv'], default='auto',
                        help="Reference file format (auto-detect, text file, or CSV)")
    parser.add_argument("--csv_filename_col", default='filename',
                        help="Column name for filenames in CSV (default: filename)")
    parser.add_argument("--csv_text_col", default='text',
                        help="Column name for transcription text in CSV (default: text)")
    parser.add_argument("--output_json", default=None,
                        help="Optional JSON file to write WER results")
    parser.add_argument("--details", action="store_true",
                        help="Print detailed error metrics")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose debugging information")
    parser.add_argument("--no_normalization", action="store_true",
                        help="Skip text normalization (use raw text)")
    args = parser.parse_args()
    
    # Check hypothesis directory
    if not os.path.exists(args.hypothesis_dir):
        print(f"Error: Hypothesis directory {args.hypothesis_dir} does not exist")
        sys.exit(1)
    
    # Load hypotheses
    hypotheses = load_hypotheses(args.hypothesis_dir, args.verbose)
    if not hypotheses:
        print(f"Error: No hypothesis files found in {args.hypothesis_dir}")
        sys.exit(1)
    
    print(f"Loaded {len(hypotheses)} hypothesis files")
    
    # Check reference file
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file {args.reference_file} does not exist")
        sys.exit(1)
    
    # Auto-detect or use specified reference format
    if args.reference_format == 'auto':
        detected_format = auto_detect_reference_format(args.reference_file)
        print(f"Auto-detected reference format: {detected_format}")
    else:
        detected_format = args.reference_format
    
    # Load references based on format
    if detected_format == 'csv':
        references = load_references_from_csv(
            args.reference_file, 
            args.csv_filename_col, 
            args.csv_text_col
        )
    else:
        references = load_references_from_text_file(args.reference_file)
    
    if not references:
        print(f"Error: No references found in {args.reference_file}")
        sys.exit(1)
    
    print(f"Loaded {len(references)} reference transcriptions from {detected_format} format")
    
    # Match hypotheses with references
    matched_pairs = match_hypotheses_with_references(hypotheses, references, args.verbose)
    
    if not matched_pairs:
        print("Error: No matching hypothesis-reference pairs found")
        print("This could be due to file naming inconsistencies.")
        if args.verbose:
            print(f"\nSample hypothesis keys: {list(hypotheses.keys())[:5]}")
            print(f"Sample reference keys: {list(references.keys())[:5]}")
        sys.exit(1)
    
    print(f"Found {len(matched_pairs)} matching hypothesis-reference pairs")
    
    # Show sample pairs
    if args.verbose or len(matched_pairs) < 5:
        print("\nSample matched pairs:")
        for i, (hyp, ref) in enumerate(matched_pairs[:3]):
            print(f"  Pair {i+1}:")
            print(f"    Reference: {ref[:50]}..." if len(ref) > 50 else f"    Reference: {ref}")
            print(f"    Hypothesis: {hyp[:50]}..." if len(hyp) > 50 else f"    Hypothesis: {hyp}")
    
    # Extract text lists
    hypotheses_list, references_list = zip(*matched_pairs)
    
    # Apply text normalization unless disabled
    if args.no_normalization:
        print("Skipping text normalization (using raw text)")
        normalized_hypotheses = list(hypotheses_list)
        normalized_references = list(references_list)
    else:
        normalizer = EnglishTextNormalizer()
        normalized_hypotheses = [normalizer(text) for text in hypotheses_list]
        normalized_references = [normalizer(text) for text in references_list]
    
    # Calculate WER
    try:
        wer = jiwer.wer(normalized_references, normalized_hypotheses)
        wer_percent = wer * 100
        
        # Quality checks
        if wer_percent >= 99.99:
            if len(set(normalized_hypotheses)) <= 2: 
                print("Warning: Almost all hypothesis transcriptions are identical.")
                print("This may indicate a problem with the model output or file processing.")
                sample_hyp = normalized_hypotheses[0]
                print(f"Sample hypothesis: {sample_hyp[:50]}...")
            
            hyp_avg_len = sum(len(h) for h in normalized_hypotheses) / len(normalized_hypotheses)
            ref_avg_len = sum(len(r) for r in normalized_references) / len(normalized_references)
            if hyp_avg_len < ref_avg_len * 0.5 or hyp_avg_len > ref_avg_len * 2:
                print(f"Warning: Significant length difference between hypotheses (avg: {hyp_avg_len:.1f} chars)")
                print(f"and references (avg: {ref_avg_len:.1f} chars).")
        
        # Print results
        if args.details:
            measures = jiwer.compute_measures(normalized_references, normalized_hypotheses)
            substitutions = measures['substitutions'] / measures['reference_length'] * 100
            deletions = measures['deletions'] / measures['reference_length'] * 100
            insertions = measures['insertions'] / measures['reference_length'] * 100
            
            print(f"\n=== WER Analysis Results ===")
            print(f"WER: {wer_percent:.2f}%")
            print(f"Word Accuracy: {100 - wer_percent:.2f}%")
            print(f"Substitution rate: {substitutions:.2f}%")
            print(f"Deletion rate: {deletions:.2f}%")
            print(f"Insertion rate: {insertions:.2f}%")
            print(f"Total word errors: {measures['substitutions'] + measures['deletions'] + measures['insertions']}")
            print(f"Reference length: {measures['reference_length']} words")
        else:
            print(f"WER: {wer_percent:.2f}%")
        
        # Save to JSON if requested
        if args.output_json:
            wer_results = {
                "wer_percent": wer_percent,
                "wer_decimal": wer,
                "word_accuracy": 100 - wer_percent,
                "total_pairs": len(matched_pairs),
                "reference_format": detected_format,
                "normalization_applied": not args.no_normalization
            }
            
            if args.details:
                measures = jiwer.compute_measures(normalized_references, normalized_hypotheses)
                wer_results.update({
                    "substitutions": measures['substitutions'],
                    "deletions": measures['deletions'],
                    "insertions": measures['insertions'],
                    "reference_length": measures['reference_length'],
                    "substitution_rate": substitutions,
                    "deletion_rate": deletions,
                    "insertion_rate": insertions,
                    "total_errors": measures['substitutions'] + measures['deletions'] + measures['insertions']
                })
            
            with open(args.output_json, 'w') as f:
                json.dump(wer_results, f, indent=2)
            
            print(f"WER results saved to {args.output_json}")
        
        return wer_percent
    
    except Exception as e:
        print(f"Error calculating WER: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()