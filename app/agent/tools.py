# tools.py — Agent 可调用的工具定义

import os
import re
import json
from pathlib import Path

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.core.logging import get_logger

logger = get_logger()

# ========== 中文城市名 → 英文映射 ==========

CITY_NAME_MAP = {
    "北京": "Beijing", "上海": "Shanghai", "天津": "Tianjin", "重庆": "Chongqing",
    "广州": "Guangzhou", "深圳": "Shenzhen", "杭州": "Hangzhou", "南京": "Nanjing",
    "武汉": "Wuhan", "成都": "Chengdu", "西安": "Xian", "郑州": "Zhengzhou",
    "沈阳": "Shenyang", "青岛": "Qingdao", "大连": "Dalian", "厦门": "Xiamen",
    "长沙": "Changsha", "苏州": "Suzhou", "哈尔滨": "Harbin", "长春": "Changchun",
    "石家庄": "Shijiazhuang", "济南": "Jinan", "福州": "Fuzhou", "合肥": "Hefei",
    "南昌": "Nanchang", "昆明": "Kunming", "贵阳": "Guiyang", "南宁": "Nanning",
    "海口": "Haikou", "太原": "Taiyuan", "兰州": "Lanzhou", "呼和浩特": "Hohhot",
    "乌鲁木齐": "Urumqi", "拉萨": "Lhasa", "银川": "Yinchuan", "西宁": "Xining",
    "台北": "Taipei", "香港": "Hong Kong", "澳门": "Macau",
    "东京": "Tokyo", "首尔": "Seoul", "纽约": "New York", "伦敦": "London",
    "巴黎": "Paris", "柏林": "Berlin", "悉尼": "Sydney", "新加坡": "Singapore",
    "曼谷": "Bangkok", "莫斯科": "Moscow",
}


def normalize_city_name(city: str) -> str:
    trimmed = city.strip()
    if trimmed in CITY_NAME_MAP:
        return CITY_NAME_MAP[trimmed]
    if re.match(r"^[a-zA-Z\s]+$", trimmed):
        return trimmed
    return trimmed


async def fetch_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "错误：未配置 OPENWEATHERMAP_API_KEY"

    city_en = normalize_city_name(city)
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city_en}&appid={api_key}&units=metric&lang=zh_cn"
    )

    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        if res.status_code != 200:
            if res.status_code == 404:
                return f'未找到城市"{city}"，请检查城市名称是否正确。'
            if res.status_code == 401:
                return "API Key 无效，请检查配置。"
            return f"天气查询失败（HTTP {res.status_code}）"

        data = res.json()
        weather = data["weather"][0]
        return "\n".join([
            f'🌍 城市：{data["name"]}（{data["sys"]["country"]}）',
            f'🌡️ 温度：{data["main"]["temp"]}°C（体感 {data["main"]["feels_like"]}°C）',
            f'☁️ 天气：{weather["description"]}',
            f'💧 湿度：{data["main"]["humidity"]}%',
            f'💨 风速：{data["wind"]["speed"]} m/s',
            f'🔽 最低温：{data["main"]["temp_min"]}°C  🔼 最高温：{data["main"]["temp_max"]}°C',
        ])


# ========== 工具定义 ==========

@tool
async def get_deepseek_info() -> str:
    """获取 DeepSeek 公司的详细介绍信息，包括公司背景、技术突破（V3/V4 系列）、架构创新（MoE、混合注意力）、开源战略（MIT 协议）、商业模式（API 定价）、融资发展等完整公司介绍。当用户询问关于 DeepSeek 公司整体情况时使用此工具。"""
    logger.info("get_deepseek_info 被调用")
    try:
        content = Path("data/documents/deepseek介绍手册.txt").read_text(encoding="utf-8")
        logger.info(f"get_deepseek_info 返回 {len(content)} 字符")
        return content
    except Exception as err:
        logger.error(f"get_deepseek_info 读取失败: {err}")
        return "无法获取 DeepSeek 公司介绍信息，请稍后重试。"


class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(description="搜索关键词或问题，应简洁明确")


@tool(args_schema=SearchKnowledgeBaseInput)
async def search_knowledge_base(query: str) -> str:
    """在知识库中搜索与用户问题相关的文档内容。当用户询问关于 DeepSeek 技术的具体细节（如模型参数、版本特性、性能指标、架构细节、技术参数等）时，调用此工具获取相关信息。输入应为简洁的关键词或问题。"""
    logger.info(f'search_knowledge_base 被调用, query: "{query}"')
    # 此函数需要 retriever 注入，实际调用时由 agent.py 传入闭包
    # 这里抛出提示，实际使用时会被 agent.py 中的工厂函数替换
    return "知识库检索功能未初始化，请通过 create_tools() 工厂函数创建工具。"


class GetWeatherInput(BaseModel):
    city: str = Field(description="城市名称，如 北京、上海、深圳、Tokyo")


@tool(args_schema=GetWeatherInput)
async def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息，包括温度、天气状况、湿度、风速等。当用户询问天气、温度、气候等实时信息时使用此工具。输入应为城市名称，如"北京"、"上海"、"Tokyo"等。"""
    logger.info(f'get_weather 被调用, city: "{city}"')
    return await fetch_weather(city)


def create_tools(retriever_getter):
    """创建所有工具（注入 retriever getter 依赖）

    Args:
        retriever_getter: 异步可调用对象，调用时返回混合检索器实例
            （调用时实时解析，可命中缓存；文档增量入库并刷新缓存后自动拿到最新实例）

    Returns:
        list: 工具列表
    """

    # Tool 1: get_deepseek_info — 直接返回，无需注入
    tool1 = get_deepseek_info

    # Tool 2: search_knowledge_base — 注入 retriever getter
    class SearchKBInput(BaseModel):
        query: str = Field(description="搜索关键词或问题，应简洁明确")

    @tool(args_schema=SearchKBInput)
    async def search_kb(query: str) -> str:
        """在知识库中搜索与用户问题相关的文档内容。当用户询问关于 DeepSeek 技术的具体细节（如模型参数、版本特性、性能指标、架构细节、技术参数等）时，调用此工具获取相关信息。输入应为简洁的关键词或问题。"""
        logger.info(f'search_knowledge_base (injected) 被调用, query: "{query}"')
        try:
            retriever = await retriever_getter()
            docs = await retriever._get_relevant_documents(query)
            result = "\n\n---\n\n".join(doc.page_content for doc in docs)
            logger.info(f"search_knowledge_base 返回 {len(docs)} 个文档片段")
            if not result.strip():
                return "知识库中没有找到相关信息。"
            return result
        except Exception as err:
            logger.error(f"search_knowledge_base 检索失败: {err}")
            return "知识库检索出错，请稍后重试。"

    # Tool 3: get_weather — 直接返回
    tool3 = get_weather

    return [tool1, search_kb, tool3]
