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
from dateutil import parser  # 需要安装 python-dateutil

# ================= 配置区域 =================
KEYWORDS = [
    '钢铁', '铁矿石', '钢材', '钢价', 'steel', 'iron ore',
    '焦炭', '钢坯', '圆钢', '方钢', '螺纹钢', '热卷', '冷轧卷',
    '钢管', '无缝管', '彩涂卷', '镀锌卷', '热轧', '钢板'
]

RSS_SOURCES = [
    # 请替换为国内可用的钢铁 RSS 源
    'https://news.google.com/rss/search?q=钢铁&hl=zh-CN&gl=CN&ceid=CN:zh-Hans',
]

NEWS_FILE = 'steel_news.json'

OUTPUT_JSON = {
    'zh': 'steel_news_zh.json',
    'en': 'steel_news_en.json',
    'fr': 'steel_news_fr.json',
    'de': 'steel_news_de.json'
}

TRANSLATION_ENABLED = True
TRANSLATE_TO = ['en', 'fr', 'de']

SIMILARITY_THRESHOLD = 0.5

# ================= 辅助函数 =================
def clean_html(raw_html):
    """彻底移除所有 HTML 标签，返回纯文本"""
    if not raw_html:
        return ''
    clean = re.sub(r'<[^>]+>', '', raw_html)
    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def parse_published(published_str):
    """尝试将发布时间字符串解析为 datetime 对象，失败则返回 None"""
    if not published_str:
        return None
    try:
        # 尝试使用 dateutil 解析各种常见格式
        return parser.parse(published_str)
    except:
        try:
            # 兼容 RFC 822 格式
            return datetime.strptime(published_str, '%a, %d %b %Y %H:%M:%S %Z')
        except:
            return None

def get_sort_key(news):
    """返回用于排序的 key，datetime 对象或原始字符串（作为后备）"""
    dt = parse_published(news.get('published', ''))
    return dt if dt else news.get('published', '')

# ================= 核心函数 =================
def fetch_rss_feeds():
    all_news = []
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.title
                raw_content = entry.get('description', entry.get('summary', ''))
                clean_content = clean_html(raw_content)
                if not clean_content:
                    clean_content = title

                if any(kw.lower() in (title + clean_content).lower() for kw in KEYWORDS):
                    img_match = re.search(r'<img.*?src="(.*?)"', raw_content)
                    image_url = img_match.group(1) if img_match else None
                    news = {
                        'title': title,
                        'link': entry.link,
                        'published': entry.published if 'published' in entry else entry.updated,
                        'source': feed.feed.title if 'title' in feed.feed else '未知',
                        'content': clean_content[:300] + ('...' if len(clean_content) > 300 else ''),
                        'image': image_url,
                        'id': hashlib.md5(entry.link.encode()).hexdigest()
                    }
                    all_news.append(news)
        except Exception as e:
            print(f"抓取 RSS 源 {url} 失败: {e}")
    return all_news

def deduplicate_news(news_list):
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
    if os.path.exists(NEWS_FILE):
        with open(NEWS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_news(news_list):
    with open(NEWS_FILE, 'w', encoding='utf-8') as f:
        json.dump(news_list, f, ensure_ascii=False, indent=2)

def merge_news(existing, new):
    existing_ids = {n['id'] for n in existing}
    unique_new = [n for n in new if n['id'] not in existing_ids]
    if not unique_new:
        return existing
    unique_new = deduplicate_news(unique_new)
    all_news = unique_new + existing
    # 按发布时间倒序（最新在前），使用解析后的 datetime 进行排序
    all_news.sort(key=get_sort_key, reverse=True)
    return all_news

def translate_text(text, target_lang):
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
    if target_lang == 'zh':
        return news
    translated = news.copy()
    translated['title'] = translate_text(news['title'], target_lang)
    if news.get('content'):
        translated['content'] = translate_text(news['content'], target_lang)
    else:
        translated['content'] = ''
    return translated

def generate_json(news_list, lang='zh'):
    limited = news_list[:50]
    output = []
    for news in limited:
        item = {
            'title': news['title'],
            'link': news['link'],
            'published': news['published'],
            'source': news.get('source', '未知'),
            'content': news.get('content', '')[:300] + ('...' if len(news.get('content', '')) > 300 else ''),
            'image': news.get('image')
        }
        output.append(item)
    return json.dumps(output, ensure_ascii=False, indent=2)

def main():
    print(f"{datetime.now()} - 开始抓取新闻...")
    new_news = fetch_rss_feeds()
    print(f"抓取到 {len(new_news)} 条新新闻")

    existing_news = load_existing_news()
    merged_news = merge_news(existing_news, new_news)
    print(f"合并后共 {len(merged_news)} 条新闻")
    save_news(merged_news)

    # 生成中文 JSON
    zh_json = generate_json(merged_news, lang='zh')
    with open(OUTPUT_JSON['zh'], 'w', encoding='utf-8') as f:
        f.write(zh_json)
    print(f"已生成中文版 JSON：{OUTPUT_JSON['zh']}")

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