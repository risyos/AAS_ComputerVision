import os
import csv
import base64
import requests
import re
from PIL import Image
from io import BytesIO
from difflib import SequenceMatcher
from tqdm import tqdm
import pandas as pd

# === KONFIGURASI ===
LM_API = "http://localhost:1234/v1/chat/completions"
IMG_DIR = r"D:\D\Riski\computervision\AAS_ComputerVision\test"
GT_CSV = r"D:\D\Riski\computervision\AAS_ComputerVision\labels\ground_truth.csv"
OUTPUT_CSV = r"D:\D\Riski\computervision\AAS_ComputerVision\results.csv"
PROMPT = "What is the license plate number shown in this image? Respond only with the plate number."

# === Fungsi: Encode gambar ke base64 ===
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# === Fungsi: Hitung CER lengkap ===
def calculate_cer_details(gt, pred):
    matcher = SequenceMatcher(None, gt, pred)
    opcodes = matcher.get_opcodes()
    S = D = I = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            S += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            D += i2 - i1
        elif tag == 'insert':
            I += j2 - j1
    N = len(gt)
    cer = round((S + D + I) / N, 3) if N > 0 else 1.0
    formula = f"CER = ({S} + {D} + {I}) / {N}"
    return cer, formula

# === Load ground truth ===
df = pd.read_csv(GT_CSV)

# === Proses setiap gambar ===
results = []

# print(f"\n🔍 Memproses {len(df)} gambar dari folder: {IMG_DIR}\n")

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    image_name = row["image"]
    ground_truth = row["ground_truth"]
    image_path = os.path.join(IMG_DIR, image_name)

    try:
        image_base64 = encode_image_to_base64(image_path)
    except Exception as e:
        pred = f"[ERROR] {e}"
        cer, formula = 1.0, "CER = (0 + 0 + 0) / 0"
        results.append([image_name, ground_truth, pred, formula, cer])
        print(f"❌ Gagal load gambar {image_path}: {e}")
        continue

    payload = {
        "model": "llava-v1.5-7b-llamafile",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(LM_API, json=payload, timeout=120)
        response.raise_for_status()
        output_raw = response.json()["choices"][0]["message"]["content"].strip()

        # Regex lebih longgar
        matches = re.findall(r'[A-Z]{1,3}\s*\d{1,5}\s*[A-Z]{0,3}', output_raw.upper())
        if matches:
            pred = matches[0].replace(" ", "").strip()
        else:
            # fallback jika regex gagal
            fallback = re.sub(r'[^A-Z0-9]', '', output_raw.upper())
            if fallback:
                print(f"⚠️  Pola tidak cocok, gunakan fallback: {fallback}")
                pred = fallback
            else:
                print(f"❗ Tidak ditemukan pola plat nomor dalam respons: {output_raw}")
                pred = "[ERROR] No plate-like text found"

    except Exception as e:
        pred = f"[ERROR] {e}"

    cer, formula = calculate_cer_details(ground_truth, pred) if not pred.startswith("[ERROR]") else (1.0, "CER = (0 + 0 + 0) / 0")

    results.append([image_name, ground_truth, pred, formula, cer])
    print(f"{image_name} => GT: {ground_truth} | Pred: {pred} | CER: {cer}")

# === Simpan ke results.csv ===
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(["image", "ground_truth", "prediction", "CER_formula", "CER_score"])
    writer.writerows(results)

print(f"\n✅ Semua prediksi selesai. Hasil disimpan di: '{OUTPUT_CSV}'")
