# message_handler.py — AG-UI 多模态消息解析与图片理解模块

import asyncio
import base64

from app.services.multimodal import describe_image, download_image
from app.core.logging import get_logger

logger = get_logger()


def strip_data_uri(value: str) -> str:
    """去除 data URI 前缀（如 data:image/png;base64,xxx → xxx），普通 base64 原样返回"""
    if isinstance(value, str) and value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def extract_text_and_images(content) -> tuple[str, list[dict]]:
    """AG-UI 用户消息 content（str 或 content part 数组）→ (文本, 图片列表)。

    图片列表元素：
      - {"bytes": bytes}  内联 base64 图片
      - {"url": str}      http(s) URL 图片

    兼容两种消息格式：
      - 新版（2025-10 生效）：{type:"image", source:{type:"data"|"url", value, mimeType?}}
      - 旧版（BinaryInputContent 兼容）：{type:"image", data:<base64>, mimeType, url?}
    """
    if isinstance(content, str):
        return content, []

    text_parts = []
    image_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text_parts.append(str(part.get("text", "")))
        elif ptype == "image":
            src = part.get("source") or {}
            stype = src.get("type")
            if stype == "data":
                raw = src.get("value")
                if raw:
                    image_parts.append({"bytes": base64.b64decode(strip_data_uri(raw))})
            elif stype == "url":
                url = src.get("value")
                if url:
                    image_parts.append({"url": url})
            elif part.get("data"):
                image_parts.append({"bytes": base64.b64decode(strip_data_uri(part["data"]))})
            elif part.get("url"):
                image_parts.append({"url": part["url"]})
    return "\n".join(text_parts).strip(), image_parts


async def describe_images(image_parts: list[dict], user_text: str) -> list[str]:
    """并发调用本地视觉模型描述多张图片。

    单张图片失败返回错误提示文本（不中断整体请求），保证文本部分照常回答。
    """

    async def _describe_one(idx: int, part: dict) -> str:
        try:
            if "bytes" in part:
                description = await describe_image(part["bytes"], user_text)
            else:
                image_bytes = await download_image(part["url"])
                description = await describe_image(image_bytes, user_text)
            if description:
                return description
            return f"（图片{idx + 1} 未能生成描述）"
        except Exception as err:
            logger.error(f"图片{idx + 1} 理解失败: {err}")
            return f"（图片{idx + 1} 理解失败：{err}）"

    return await asyncio.gather(*[_describe_one(i, p) for i, p in enumerate(image_parts)])


def compose_multimodal_message(text: str, descriptions: list[str]) -> str:
    """组装最终纯文本 user 消息：用户文字 + 各图片描述（图片→文字化）"""
    parts = []
    if text and text.strip():
        parts.append(text.strip())
    for i, desc in enumerate(descriptions):
        parts.append(f"[图片{i + 1}]: {desc}")
    return "\n\n".join(parts)
