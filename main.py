import json
import os
import random
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env")

# ---------- 加载配置 ----------
CARDS_FILE = Path(__file__).parent / "cards.json"
SPREADS_FILE = Path(__file__).parent / "spreads.json"
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 加载牌库
_cards_cache: list[dict] | None = None

def load_cards() -> list[dict]:
    global _cards_cache
    if _cards_cache is None:
        with open(CARDS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _cards_cache = data["cards"]
    return _cards_cache

# 加载牌阵配置
_spreads_cache: list[dict] | None = None

def load_spreads() -> list[dict]:
    global _spreads_cache
    if _spreads_cache is None:
        with open(SPREADS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _spreads_cache = data["spreads"]
    return _spreads_cache

def get_spread(spread_id: str) -> dict | None:
    for spread in load_spreads():
        if spread["id"] == spread_id:
            return spread
    return None

SYSTEM_PROMPT = """你是一位温柔、有洞察力的塔罗师。请根据用户的提问、抽到的牌及其正逆位，给出温暖而有力量的解读。

解读要求：
- 正位牌：侧重启示与可行建议，帮助用户看见方向与可能。
- 逆位牌：侧重需要留意的调整与成长空间，语气温和，不制造恐惧或灾难感。
- 结合牌阵整体脉络，回应用户的具体问题。
- 每次解读控制在 3-5 句话，语言亲切、清晰、有支持感。"""

app = FastAPI(title="塔罗 AI 服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 数据模型 ----------

class DrawRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    spread: str = Field(default="single", description="牌阵ID")

class ShuffleRequest(BaseModel):
    seed: int | None = Field(default=None, description="随机种子")

class ShuffledCard(BaseModel):
    index: int
    card_id: int

class ShuffleResponse(BaseModel):
    total: int
    shuffled: list[ShuffledCard]

class DrawnCard(BaseModel):
    name: str
    name_en: str
    orientation: Literal["upright", "reversed"]
    orientation_label: str
    keywords: list[str]
    meaning: str
    position: str | None = None

class InterpretRequest(BaseModel):
    question: str = Field(..., min_length=1)
    spread: str = Field(default="single")
    selected_cards: list[int] = Field(..., description="用户选中的牌ID列表")
    orientations: list[str] = Field(default=[], description="每张牌的正逆位")

class InterpretResponse(BaseModel):
    spread: str
    spread_name: str
    question: str
    cards: list[DrawnCard]
    interpretation: str

# ---------- 核心逻辑 ----------

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")

@app.get("/card-back.png")
async def card_back():
    return FileResponse(Path(__file__).parent / "card-back.png", media_type="image/png")

@app.get("/spreads")
async def get_spreads():
    """获取所有可用牌阵"""
    return load_spreads()

@app.post("/shuffle", response_model=ShuffleResponse)
async def shuffle(request: ShuffleRequest):
    """洗牌：返回78张牌的随机顺序"""
    deck = load_cards()
    total = len(deck)
    
    rng = random.Random(request.seed) if request.seed else random
    indices = list(range(total))
    rng.shuffle(indices)
    
    shuffled = [
        ShuffledCard(index=i, card_id=deck[idx]["id"])
        for i, idx in enumerate(indices)
    ]
    
    return ShuffleResponse(total=total, shuffled=shuffled)

@app.post("/interpret", response_model=InterpretResponse)
async def interpret(request: InterpretRequest):
    """解读：根据用户选中的牌生成解读"""
    spread = get_spread(request.spread)
    if not spread:
        raise HTTPException(status_code=400, detail=f"不支持的牌阵：{request.spread}")
    
    deck = load_cards()
    deck_by_id = {card["id"]: card for card in deck}
    
    if len(request.selected_cards) != spread["cardCount"]:
        raise HTTPException(
            status_code=400,
            detail=f"牌阵{spread['name']}需要{spread['cardCount']}张牌"
        )
    
    drawn = []
    positions = spread["positions"]
    
    for i, card_id in enumerate(request.selected_cards):
        card = deck_by_id.get(card_id)
        if not card:
            raise HTTPException(status_code=400, detail=f"无效的牌ID：{card_id}")
        
        if i < len(request.orientations):
            is_upright = request.orientations[i] == "upright"
        else:
            is_upright = random.choice([True, False])
        
        orientation = "upright" if is_upright else "reversed"
        item = {
            "name": card["name"],
            "name_en": card["nameEn"],
            "orientation": orientation,
            "orientation_label": "正位" if is_upright else "逆位",
            "keywords": card["keywords"],
            "meaning": card["upright"] if is_upright else card["reversed"],
        }
        if i < len(positions):
            item["position"] = positions[i]
        drawn.append(item)
    
    # 构建Prompt并调用DeepSeek
    lines = [
        f"用户问题：{request.question}",
        f"牌阵：{spread['name']} - {spread['description']}",
        "",
        "抽到的牌：",
    ]
    for card in drawn:
        pos = f"（{card['position']}）" if card.get("position") else ""
        keywords = "、".join(card["keywords"])
        lines.append(
            f"- {card['name']}（{card['name_en']}）{pos} · {card['orientation_label']}\n"
            f"  关键词：{keywords}\n"
            f"  牌意：{card['meaning']}"
        )
    lines.append("")
    lines.append("请结合牌阵的整体脉络，给出温暖而有力量的解读。")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")
    
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "temperature": 0.8,
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"DeepSeek API 错误：{exc.response.text}")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"连接失败：{exc}")
    
    data = response.json()
    try:
        interpretation = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="API 返回格式异常")
    
    return InterpretResponse(
        spread=request.spread,
        spread_name=spread["name"],
        question=request.question,
        cards=[DrawnCard(**card) for card in drawn],
        interpretation=interpretation,
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "model": DEEPSEEK_MODEL,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
