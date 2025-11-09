import re
import cv2
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
import numpy as np
from typing import Dict, Optional

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

CBC_KEYS = {
    "HGB": ["hemoglobin", "hgb", "hb"],
    "WBC": ["total leukocyte", "tlc", "wbc", "leukocyte count", "white blood cell", "white cell", "white blood count"],
    "RBC": ["rbc", "rbc count", "red blood cell", "red cell", "erythrocyte", "erythrocyte count"],
    "PLT": ["platelet","platelets", "platelet count", "plt", "plt count", "thrombocyte", "thrombocyte count"],
    "HCT": ["hematocrit", "pcv", "hct"],
    "MCV": ["mcv", "mean corpuscular volume"],
    "MCH": ["mch ","Mean Corpuscular Hemoglobin"],
    "MCHC": ["mchc","m c h c","m-ch-c","mean corpuscular hemoglobin concentration"],
    "RDWCV": ["rdwcv", "rdw-cv", "rdw cv"],
    "RDWSD": ["rdw-sd", "rdw sd","rdwsd"],}

VALID = {
    "HGB": (5, 20),
    "WBC": (2, 20),
    "RBC": (3, 7),
    "PLT": (50, 700),
    "HCT": (20, 60),
    "MCV": (60, 120),
    "MCH": (20, 40),
    "MCHC": (25, 40),
    "RDWSD": (20, 80),
    "RDWCV": (8, 25),
}

def preprocess(path):
    img = cv2.imread(path)

    # resize to high DPI
    h = 2500
    scale = h / img.shape[0]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # heavy denoise
    gray = cv2.bilateralFilter(gray, 15, 75, 75)

    # sharpen
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # high contrast threshold
    th = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31, 6)
    return th

def get_text(path):
    img = preprocess(path)
    config = "--oem 3 --psm 4" 
    text = pytesseract.image_to_string(img, config=config)
    return text

def extract_number(line):
    nums = re.findall(r"\d+\.?\d*", line)
    if not nums:
        return None
    return float(nums[0])

def extract_cbc(text):
    raw = text
    text = raw.lower()
    text = re.sub(r"[:|]+", " ", text)
    text = text.replace("-", " ").replace("_", " ").replace("/", " ")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    extracted = {
        "HGB": None, "WBC": None, "RBC": None, "PLT": None, "HCT": None,
        "MCV": None, "MCH": None, "MCHC": None, "RDWSD": None, "RDWCV": None
    }

    # NORMAL CBC FIELDS (already working fine)
    patt = {
        "HGB":   r"\b(hgb|hemoglobin)\b",
        "WBC":   r"\b(wbc|leukocyte|tlc|white\s*blood\s*(cell|count)?)\b",
        "RBC":   r"\brbc\b",
        "PLT":   r"\b(plt|platelet|platelet count|platelets)\b",
        "HCT":   r"\b(hct|hematocrit|pcv)\b",
        "MCV":   r"\bmcv\b",
        "MCHC":  r"\bmchc\b",
        "MCH":   r"(?<![a-z])mch(?!c)(?!\s*c)\b",
    }

    def first_number(s):
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else None

    # PASS 1 — Standard Matches
    for field, rx in patt.items():
        for line in lines:
            if re.search(rx, line):
                num = first_number(line)
                if num is not None:
                    if field == "WBC" and num > 100:
                        num = round(num / 1000.0, 2)
                    lo, hi = VALID[field]
                    if lo <= num <= hi:
                        extracted[field] = num
                break

    fuzzy_rdw = ["rdw", "raw", "row", "rdm", "ndw", "pdw", "rd w", "r dw"]

    def line_is_rdw_like(s):
        s = s.replace(" ", "")
        for fr in fuzzy_rdw:
            if fuzz.partial_ratio(fr, s) > 80:
                return True
        return False

    # PASS 2A — Scan lines that look like RDW rows
    for line in lines:
        if line_is_rdw_like(line):
            if any(x in line for x in ["cv", "c v", "v "]):
                num = first_number(line)
                if num and VALID["RDWCV"][0] <= num <= VALID["RDWCV"][1]:
                    extracted["RDWCV"] = num

            if any(x in line for x in ["sd", "s d"]):
                num = first_number(line)
                if num and VALID["RDWSD"][0] <= num <= VALID["RDWSD"][1]:
                    extracted["RDWSD"] = num
    if extracted["RDWCV"] is None or extracted["RDWSD"] is None:
        for line in lines:
            nums = re.findall(r"\d+\.?\d*", line)
            nums = [float(n) for n in nums]

            for n in nums:
                if extracted["RDWCV"] is None and 8 <= n <= 25:
                    extracted["RDWCV"] = n
                if extracted["RDWSD"] is None and 30 <= n <= 80:
                    extracted["RDWSD"] = n

    return extracted


def extract_from_image_or_pdf(path):
    text = get_text(path)
    print(text)
    data = extract_cbc(text)

    print("\nExtracted CBC Values:")
    for k,v in data.items():
        print(f"{k}: {v}")
    return data