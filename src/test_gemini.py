import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ----------------------------------------
# 1. 找到项目根目录
# ----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------
# 2. 加载 .env
# ----------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


# ----------------------------------------
# 3. 读取 Gemini API Key
# ----------------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")


# ----------------------------------------
# 4. 创建 Gemini Client
# ----------------------------------------
client = genai.Client(api_key=api_key)


# ----------------------------------------
# 5. 调用 Gemini
# ----------------------------------------
interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="Reply with exactly: Gemini connection OK"
)


# ----------------------------------------
# 6. 输出结果
# ----------------------------------------
print(interaction.output_text)