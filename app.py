"""
Multimodal Image Question-Answering API
-----------------------------------------
POST /answer-image
Body: {"image_base64": "...", "question": "..."}
Response: {"answer": "..."}

Uses a vision-capable model (gpt-4o-mini) via AI Pipe.
Environment variable required: AIPIPE_TOKEN
"""

import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# ---- Enable CORS for all origins (required so the grader's Cloudflare Worker can call us) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"


class ImageQARequest(BaseModel):
    image_base64: str
    question: str


class ImageQAResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/answer-image", response_model=ImageQAResponse)
async def answer_image(req: ImageQARequest):
    if not AIPIPE_TOKEN:
        raise HTTPException(status_code=500, detail="AIPIPE_TOKEN not set on server")

    # Some graders send raw base64, others send a full data URL - handle both
    img_data = req.image_base64
    if not img_data.startswith("data:image"):
        img_data = f"data:image/png;base64,{img_data}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{req.question}\n\n"
                            "Answer with ONLY the raw value. If the answer is a number, "
                            "return just the number with no currency symbols, units, or commas."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": img_data}},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(CHAT_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"LLM API error: {resp.status_code} {resp.text}"
        )

    data = resp.json()
    answer_text = data["choices"][0]["message"]["content"].strip()

    return ImageQAResponse(answer=answer_text)
