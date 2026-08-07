# parse_pdf.py — PDF 混合解析：PyMuPDF 文本提取 + 页面渲染 + RapidOCR 识别图片文字

import numpy as np
import fitz  # PyMuPDF

# 文本提取的最小字符数 —— 低于此值认为 PDF 以图片为主，优先使用 OCR 结果
MIN_TEXT_LENGTH = 80

# OCR 引擎单例（模型加载开销大，复用避免重复加载）
_ocr_engine = None


def _get_ocr_engine():
    """惰性初始化并缓存 RapidOCR 引擎"""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _page_to_image(page, scale: float = 2.0) -> np.ndarray:
    """将 PDF 页面渲染为 numpy 图像数组（RapidOCR 输入格式）"""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    # RGBA → RGB（RapidOCR 需要 3 通道）
    if pix.n == 4:
        img = img[:, :, :3]
    return img


def _ocr_page(img: np.ndarray) -> str:
    """对单页渲染图像执行 OCR，返回识别的文本"""
    output = _get_ocr_engine()(img)
    # rapidocr 3.x 返回 RapidOCROutput 对象；2.x 返回 (result, elapse) 元组
    if isinstance(output, tuple):
        result = output[0] or []
        return "\n".join(item[1] for item in result).strip()
    txts = output.txts or ()
    return "\n".join(txts).strip()


def parse_pdf(file_path: str) -> str:
    """解析 PDF 文件（混合模式）

    策略：
    1. 用 PyMuPDF 提取可选择文本（适合文字型 PDF）
    2. 再用 PyMuPDF 渲染每页为图片，RapidOCR 识别图片文字
    3. 智能合并两路结果

    Args:
        file_path: PDF 文件路径

    Returns:
        提取的文本内容
    """
    doc = fitz.open(file_path)
    num_pages = len(doc)
    print(f"  [PDF] 共 {num_pages} 页")

    text_parts = []
    ocr_parts = []

    for i in range(num_pages):
        print(f"  [PDF] 处理第 {i + 1}/{num_pages} 页...")
        page = doc[i]

        # ---- 提取可选择文本 ----
        try:
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
                print(f"   文本: {len(page_text)} 字符")
        except Exception as text_err:
            print(f"   文本提取失败: {text_err}")

        # ---- 渲染为图片 + OCR ----
        try:
            img = _page_to_image(page, scale=2.0)  # 2x 提升 OCR 精度
            height, width = img.shape[:2]
            if width < 10 or height < 10:
                print(f"   页面尺寸过小 ({width}x{height})，跳过 OCR")
                continue

            ocr_page_text = _ocr_page(img)
            if ocr_page_text:
                ocr_parts.append(ocr_page_text)
                print(f"   OCR: {len(ocr_page_text)} 字符")
        except Exception as ocr_err:
            print(f"   OCR 失败: {ocr_err}")

    doc.close()

    # ---- 智能合并 ----
    text_content = "\n".join(text_parts)
    ocr_text = "\n\n".join(ocr_parts)
    merged = merge_text(text_content, ocr_text)
    print(f"  [PDF] 合并后共 {len(merged)} 字符")
    return merged


def merge_text(text_content: str, ocr_text: str) -> str:
    """智能合并文本提取与 OCR 文本"""
    text_len = len(text_content or "")
    ocr_len = len(ocr_text or "")

    if text_len >= MIN_TEXT_LENGTH and ocr_len > 0:
        # 文字型 PDF + 含图片：合并两者，用分隔符标明
        return text_content + "\n\n--- 以下为图片 OCR 内容 ---\n" + ocr_text
    elif text_len >= MIN_TEXT_LENGTH:
        # 文字型 PDF，无 OCR 内容
        return text_content
    elif ocr_len > 0:
        # 图片型 PDF（扫描件），优先使用 OCR 结果
        return ocr_text
    else:
        # 两路均失败，返回有内容的那一路（可能为空）
        return text_content or ocr_text or ""
