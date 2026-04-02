"""Manual probe: ask GPT-5.4 specific questions about a game frame."""
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import os

client = OpenAI(
    base_url=os.environ["AZURE_FOUNDRY_BASE_URL"],
    api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
)
model = os.environ.get("AZURE_MODEL", "gpt-5.4")

# Load the initial frame
frame_path = Path("experiments/E001-baseline/frames/ls20-9607627b/step_000_initial.png")
with open(frame_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

questions = [
    "Describe EVERY distinct object you see in this image. For each object, tell me: what color it is, where it is (top/bottom/left/right/center), approximate size, and what shape it has. Be exhaustive.",
    "This is a game. Looking at this image, which object do you think the player controls (the avatar)? Why? What visual clues tell you this?",
    "Do you see anything that looks like a health bar, progress bar, or life indicator? Describe it in detail — colors, position, what it might represent.",
    "If you had to guess what the GOAL of this game is just from looking at this single frame, what would you guess? What visual elements inform your guess?",
    "I see a small icon in the top-center inside a dark box. What do you think it represents? Also, there seems to be a vertical line connecting it to the main area — what might that mean?",
]

for i, q in enumerate(questions):
    print(f"\n{'='*70}")
    print(f"QUESTION {i+1}: {q[:80]}...")
    print('='*70)
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": q},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
            ],
        }],
        temperature=0.3,
        max_completion_tokens=800,
    )
    answer = resp.choices[0].message.content
    print(f"\nANSWER:\n{answer}\n")

print("\n=== MANUAL PROBE COMPLETE ===")
