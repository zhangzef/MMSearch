import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageDraw, ImageFont
import textwrap
import re


def save_url_content_as_image(
    url: str,
    output_path: str = "output.png",
    width: int = 800,
    font_size: int = 16,
    max_chars_per_line: int = 60,
    margin: int = 20,
):
    """
    从 URL 加载内容并转为图片保存在本地。

    - 如果 URL 是图片地址：直接下载保存。
    - 如果 URL 是网页：提取文本内容并绘制成图片。

    :param url: 要加载的 URL
    :param output_path: 输出图片路径
    :param width: 图片宽度
    :param font_size: 字体大小
    :param max_chars_per_line: 每行最多字符数（用于简单换行）
    :param margin: 图片边距
    :return: 保存的图片路径
    """

    # 配置带重试的 session，缓解偶发网络/SSL 问题
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.SSLError as e:
        # 这里你也可以选择改成 log 或者自定义异常
        print(f"SSL 错误：{e}")
        raise
    except requests.RequestException as e:
        print(f"请求失败：{e}")
        raise

    content_type = resp.headers.get("Content-Type", "")

    # 情况 1：如果本来就是图片，直接保存
    if content_type.startswith("image/"):
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"已将图片保存到：{output_path}")
        return output_path

    # 情况 2：网页/文本 -> 渲染为一张文字图片
    # 简单去掉 HTML 标签（不完美，但够用）
    html = resp.text
    text = re.sub(r"<[^>]+>", "", html)  # 去掉标签
    text = re.sub(r"\s+\n", "\n", text)  # 清理多余空白
    text = text.strip()

    if not text:
        text = "[页面内容为空或无法解析文本]"

    # 按字符数粗暴换行（简单实用型）
    wrapped_lines = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            wrapped_lines.append("")  # 空行
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=max_chars_per_line))

    # 字体（使用默认字体，跨平台不挑）
    font = ImageFont.load_default()

    line_height = font_size + 6  # 行高，留一点行距
    height = margin * 2 + line_height * len(wrapped_lines)
    if height < 200:
        height = 200  # 最少给点高度，免得太小

    # 创建白底图片
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    y = margin
    for line in wrapped_lines:
        draw.text((margin, y), line, font=font, fill="black")
        y += line_height

    img.save(output_path)
    print(f"已将页面文本渲染成图片保存到：{output_path}")
    return output_path
