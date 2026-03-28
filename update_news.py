#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import feedparser
import requests
import json
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# ================= 配置区域 =================
# 关键词列表（中英文）
KEYWORDS = [
    '钢铁', '铁矿石', '钢材', '钢价', 'steel', 'iron ore',
    '焦炭', '钢坯', '圆钢', '方钢', '螺纹钢', '热卷', '冷轧卷',
    '钢管', '无缝管', '彩涂卷', '镀锌卷', '热轧', '钢板'
]

# RSS 源列表（可自行增删）
RSS_SOURCES = [
    'https://news.google.com/rss/search?q=钢铁&hl=zh-CN&gl=CN&ceid=CN:zh-Hans',
    # 可添加其他 RSS 源，例如：
    # 'https://www.mysteel.com/rss/news.xml',
    # 'https://www.lgmi.com/rss/news.xml',
]

# 本地存储文件（保留原始数据，用于去重）
NEWS_FILE = 'steel_news.json'

# 输出 JSON 文件名（多语言）
OUTPUT_JSON = {
    'zh': 'steel_news_zh.json',
    'en': 'steel_news_en.json',
    'fr': 'steel_news_fr.json',
    'de': 'steel_news_de.json'
}

# 翻译开关
TRANSLATION_ENABLED = True              # 是否启用翻译
TRANSLATE_TO = ['en', 'fr', 'de']       # 目标语言代码

# 相似度去重阈值（0-1，越大越宽松）
SIMILARITY_THRESHOLD = 0.5

# ================= 核心函数 =================
def fetch_rss_feeds():
    """从所有 RSS 源抓取新闻"""
    all_news = []
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.title
                summary = entry.summary if 'summary' in entry else ''
                # 检查是否包含关键词
                if any(kw.lower() in (title + summary).lower() for kw in KEYWORDS):
                    # 提取图片
                    img_match = re.search(r'<img.*?src="(.*?)"', summary)
                    image_url = img_match.group(1) if img_match else None
                    news = {
                        'title': title,
                        'link': entry.link,
                        'published': entry.published if 'published' in entry else entry.updated,
                        'source': feed.feed.title if 'title' in feed.feed else '未知',
                        'content': summary,
                        'image': image_url,
                        'id': hashlib.md5(entry.link.encode()).hexdigest()
                    }
                    all_news.append(news)
        except Exception as e:
            print(f"抓取 RSS 源 {url} 失败: {e}")
    return all_news

def deduplicate_news(news_list):
    """基于标题相似度去重"""
    if len(news_list) <= 1:
        return news_list
    titles = [n['title'] for n in news_list]
    vectorizer = TfidfVectorizer().fit_transform(titles)
    vectors = vectorizer.toarray()
    to_keep = []
    for i in range(len(vectors)):
        dup = False
        for j in range(i):
            sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            if sim > SIMILARITY_THRESHOLD:
                dup = True
                break
        if not dup:
            to_keep.append(i)
    return [news_list[i] for i in to_keep]

def load_existing_news():
    """加载已保存的新闻"""
    if os.path.exists(NEWS_FILE):
        with open(NEWS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_news(news_list):
    """保存新闻到 JSON 文件"""
    with open(NEWS_FILE, 'w', encoding='utf-8') as f:
        json.dump(news_list, f, ensure_ascii=False, indent=2)

def merge_news(existing, new):
    """合并新新闻，去重（基于 ID 和相似度）"""
    existing_ids = {n['id'] for n in existing}
    unique_new = [n for n in new if n['id'] not in existing_ids]
    if not unique_new:
        return existing
    # 对新新闻进行相似度去重（避免同一次抓取内重复）
    unique_new = deduplicate_news(unique_new)
    # 合并并按发布时间倒序（最新在前）
    all_news = unique_new + existing
    all_news.sort(key=lambda x: x['published'], reverse=True)
    return all_news

def translate_text(text, target_lang):
    """使用 MyMemory 免费翻译 API 翻译文本"""
    if not text or target_lang == 'zh':
        return text
    if len(text) > 500:
        text = text[:500]
    url = f"https://api.mymemory.translated.net/get?q={requests.utils.quote(text)}&langpair=zh|{target_lang}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('responseStatus') == 200:
            return data['responseData']['translatedText']
    except Exception as e:
        print(f"翻译失败: {e}")
    return text

def translate_news(news, target_lang):
    """翻译单条新闻（标题和内容）"""
    if target_lang == 'zh':
        return news
    translated = news.copy()
    translated['title'] = translate_text(news['title'], target_lang)
    # 翻译正文（简单剥离 HTML 标签）
    if news.get('content'):
        clean_content = re.sub('<[^<]+?>', '', news['content'])
        translated['content'] = translate_text(clean_content, target_lang)
    else:
        translated['content'] = ''
    return translated

def generate_json(news_list, lang='zh'):
    """生成指定语言的 JSON 内容"""
    # 只保留最近 50 条，避免文件过大
    limited = news_list[:50]
    output = []
    for news in limited:
        item = {
            'title': news['title'],
            'link': news['link'],
            'published': news['published'],
            'source': news.get('source', '未知'),
            'content': news.get('content', '')[:300] + '...' if len(news.get('content', '')) > 300 else news.get('content', ''),
            'image': news.get('image')
        }
        output.append(item)
    return json.dumps(output, ensure_ascii=False, indent=2)

# ================= 主流程 =================
def main():
    print(f"{datetime.now()} - 开始抓取新闻...")

    # 1. 抓取新新闻
    new_news = fetch_rss_feeds()
    print(f"抓取到 {len(new_news)} 条新新闻")

    # 2. 加载已有新闻
    existing_news = load_existing_news()

    # 3. 合并去重
    merged_news = merge_news(existing_news, new_news)
    print(f"合并后共 {len(merged_news)} 条新闻")

    # 4. 保存主数据（用于下次去重）
    save_news(merged_news)

    # 5. 生成中文版 JSON
    zh_json = generate_json(merged_news, lang='zh')
    with open(OUTPUT_JSON['zh'], 'w', encoding='utf-8') as f:
        f.write(zh_json)
    print(f"已生成中文版 JSON：{OUTPUT_JSON['zh']}")

    # 6. 多语言翻译与生成
    if TRANSLATION_ENABLED:
        for lang in TRANSLATE_TO:
            print(f"正在翻译为 {lang}...")
            translated_news = [translate_news(n, lang) for n in merged_news]
            lang_json = generate_json(translated_news, lang=lang)
            with open(OUTPUT_JSON[lang], 'w', encoding='utf-8') as f:
                f.write(lang_json)
            print(f"已生成 {lang} 版 JSON：{OUTPUT_JSON[lang]}")

    print("所有任务完成！")

if __name__ == '__main__':
    main()