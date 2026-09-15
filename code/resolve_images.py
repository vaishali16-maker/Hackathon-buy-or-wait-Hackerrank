"""One-off script to resolve missing amounts from linked images for
request_16 (event_1442) and request_20 (event_1786), using Groq's
vision-capable model.
"""

import base64

import os
import pandas as pd
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

TARGET_EVENT_IDS = ["event_1442", "event_1786"]

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    images_df = pd.read_csv("dataset/images.csv")
    relevant = images_df[images_df["related_event_id"].isin(TARGET_EVENT_IDS)]

    if relevant.empty:
        print("No matching images found — check images.csv column names.")
        print(images_df.head())
        return

    for _, row in relevant.iterrows():
        image_id = row["image_id"]
        event_id = row["related_event_id"]
        image_path = f"dataset/media/images/{image_id}.png"

        if not os.path.exists(image_path):
            print(f"[missing file] {image_path}")
            continue

        b64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract ONLY the final total monetary amount from this financial "
                                "document (bill, invoice, payslip, or statement). Respond with "
                                "nothing except the plain number — no currency symbol, no commas, "
                                "no explanation, no reasoning, no thinking out loud. Just the number, "
                                "e.g.: 1250.50"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=600,
        )

        raw = response.choices[0].message.content.strip()
        if "</think>" in raw:
            extracted = raw.split("</think>")[-1].strip()
        else:
            extracted = raw
        print(f"event_id={event_id} | image_id={image_id} | extracted_amount={extracted}")

if __name__ == "__main__":
    main()