# multimodal.py — 云端 GLM-4V 视觉模型封装（用户上传图片理解）

# 思路：将用户上传的图片交给智谱 GLM-4V（默认 glm-4v-plus）生成文字描述，
# 描述与用户文字合并后进入现有纯文本 Agent/RAG 链路，实现"图片 → 文字化"的多模态查询。

import os
import base64
import io

import httpx
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ========== 配置（环境变量，智谱 GLM-4V 云端接口） ==========

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
# GLM-4V 对话补全接口（OpenAI 兼容格式，Bearer 鉴权）
GLM4V_BASE_URL = os.getenv(
    "GLM4V_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions"
)
# 模型可选：glm-4v-flash（免费，推荐）、glm-4v-plus（付费，需账号有余额）
# 注意：glm-4v-flash 的 max_tokens 上限为 1024，超出会报 400 (code 1210)
GLM4V_MODEL = os.getenv("GLM4V_MODEL", "glm-4v-flash")

# 图片描述超时（云端 GPU 推理通常 1-3s，留足余量）
IMAGE_DESCRIBE_TIMEOUT = 120

# 通过 URL 下载图片的最大大小（10MB），防止恶意/异常大图打爆内存
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

# 送 VLM 前把图片最长边缩放到该像素值内（降低 base64 体积与 API 传输成本）
VLM_MAX_IMAGE_EDGE = 1024

# 图片描述 Prompt：引导 VLM 输出详尽、结构化的中文描述，便于后续检索与问答引用
IMAGE_DESCRIBE_PROMPT = (
    "请仔细观察这张图片，用中文详细描述其中的内容：\n"
    "- 若是图表/表格：提取关键数据、坐标轴含义、趋势结论；\n"
    "- 若是文档/截图：完整转述其中的文字内容与版面结构；\n"
    "- 若是实物/场景图：描述主体、关系与细节。\n"
    "描述应完整详细，便于后续知识检索与问答引用。"
)


def _clean_description(text: str) -> str:
    """清洗 VLM 返回的描述文本（去首尾空白，保持内容原样）"""
    if not text:
        return ""
    return text.strip()


def _resize_image(image_bytes: bytes) -> bytes:
    """将图片等比缩放到最长边不超过 VLM_MAX_IMAGE_EDGE（减小 VLM 处理开销）。

    大图（手机截图等）在 CPU 上会生成数千个视觉 token，单次推理可达数分钟；
    缩放后 token 数量锐减，推理速度大幅提升，且不影响识别效果。
    解码失败时原样返回，交给 VLM 侧容错。
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        width, height = img.size
        longest = max(width, height)
        if longest <= VLM_MAX_IMAGE_EDGE:
            return image_bytes  # 小图直接使用原图

        scale = VLM_MAX_IMAGE_EDGE / longest
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        img = img.convert("RGB").resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
    except Exception:
        return image_bytes


def _detect_image_mime(image_bytes: bytes) -> str:
    """用 PIL 探测图片真实格式，返回 MIME（用于 data URI 前缀）"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            fmt = (img.format or "JPEG").lower()
        return "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
    except Exception:
        return "image/jpeg"


async def describe_image(image_bytes: bytes, user_text: str = "") -> str:
    """调用智谱 GLM-4V 云端视觉模型，将图片字节转为文字描述。

    Args:
        image_bytes: 图片二进制内容
        user_text: 用户附带的问题文本（可选，作为上下文提示 VLM）

    Returns:
        str: 图片的文字描述；调用失败时抛出异常由调用方兜底
    """
    if not ZHIPU_API_KEY:
        raise ValueError(
            "未配置 ZHIPU_API_KEY，请在 .env 中填入智谱开放平台（open.bigmodel.cn）的 API Key"
        )

    # 缩放后转 base64 data URI 交给云端模型（减小传输体积）
    resized = _resize_image(image_bytes)
    image_b64 = base64.b64encode(resized).decode("utf-8")
    image_data_uri = f"data:{_detect_image_mime(resized)};base64,{image_b64}"

    # 携带用户问题有助于 VLM 聚焦回答
    content = IMAGE_DESCRIBE_PROMPT
    if user_text and user_text.strip():
        content += f"\n\n用户问题：{user_text.strip()}"

    payload = {
        "model": GLM4V_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
        "max_tokens": 1024,  # glm-4v-flash 上限为 1024，超出会报 400 (code 1210)
        "temperature": 0.1,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=IMAGE_DESCRIBE_TIMEOUT) as client:
        resp = await client.post(GLM4V_BASE_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            # 带上响应体，便于从日志直接定位智谱返回的真实错误码
            raise RuntimeError(
                f"GLM-4V 调用失败 (HTTP {resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()

    try:
        description = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise RuntimeError(f"GLM-4V 响应格式异常: {data}") from err

    return _clean_description(description if isinstance(description, str) else str(description))


async def download_image(url: str) -> bytes:
    """下载 http(s) URL 指向的图片内容。

    带超时与大小上限保护，防止外部 URL 拖慢请求或打爆内存。

    Args:
        url: 图片 http(s) URL

    Returns:
        bytes: 图片二进制内容
    """
    async with httpx.AsyncClient(timeout=IMAGE_DESCRIBE_TIMEOUT, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            chunks = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"图片下载超过大小上限（{MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB）"
                    )
                chunks.append(chunk)
    return b"".join(chunks)
