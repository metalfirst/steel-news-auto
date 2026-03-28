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
KEYWORDS = [
    '钢铁', '铁矿石', '钢材', '钢价', 'steel', 'iron ore',
    '焦炭', '钢坯', '圆钢', '方钢', '螺纹钢', '热卷', '冷轧卷',
    '钢管', '无缝管', '彩涂卷', '镀锌卷', '热轧', '钢板'
]

RSS_SOURCES = [
    'https://news.google.com/rss/search?q=钢铁&hl=zh-CN&gl=CN&ceid=CN:zh-Hans',
    # 可添加更多 RSS 源
]

NEWS_FILE = 'steel_news.json'          # 本地存储所有新闻
HTML_TEMPLATE = 'template.html'        # 模板文件
OUTPUT_FILES = {
    'zh': 'steelnews.html',
    'en': 'steelnews.en.html',
    'fr': 'steelnews.fr.html',
    'de': 'steelnews.de.html'
}

TRANSLATION_ENABLED = True              # 是否启用翻译
TRANSLATE_TO = ['en', 'fr', 'de']       # 目标语言

SIMILARITY_THRESHOLD = 0.5              # 去重阈值

# HTTP 上传配置（从环境变量读取）
UPLOAD_URL = os.environ.get('UPLOAD_URL', '')
UPLOAD_TOKEN = os.environ.get('UPLOAD_TOKEN', '')

# ================= 核心功能 =================
def fetch_rss_feeds():
    """从所有RSS源抓取新闻"""
    all_news = []
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.title
                summary = entry.summary if 'summary' in entry else ''
                # 检查关键词
                if any(kw.lower() in (title + summary).lower() for kw in KEYWORDS):
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
            print(f"抓取RSS源 {url} 失败: {e}")
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
    all_news.sort(key=lambda x: x['published'], reverse=True)
    return all_news

def translate_text(text, target_lang):
    """使用 MyMemory 免费翻译 API"""
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
        clean_content = re.sub('<[^<]+?>', '', news['content'])
        translated['content'] = translate_text(clean_content, target_lang)
    else:
        translated['content'] = ''
    return translated

def generate_html(news_list, lang='zh'):
    """生成HTML，使用模板文件"""
    with open(HTML_TEMPLATE, 'r', encoding='utf-8') as f:
        template = f.read()
    
    news_items = []
    for news in news_list[:50]:  # 只显示最近50条
        published = news.get('published', '')
        try:
            dt = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
            date_str = dt.strftime('%Y-%m-%d %H:%M')
        except:
            date_str = published
        
        title = news['title']
        link = news['link']
        source = news.get('source', '未知')
        content = news.get('content', '')[:300] + '...' if len(news.get('content', '')) > 300 else news.get('content', '')
        
        image_html = ''
        if news.get('image'):
            image_html = f'''
            <div class="news-image">
                <img src="{news['image']}" alt="新闻配图" loading="lazy">
                <div class="news-image-caption">图片来源于 {source}</div>
            </div>
            '''
        
        item = f'''
        <div class="news-item">
            <div class="news-date">{date_str}</div>
            <div class="news-title"><a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></div>
            <div class="news-content">{content}</div>
            {image_html}
            <div class="news-source">来源：{source}</div>
        </div>
        <hr class="news-divider">
        '''
        news_items.append(item)
    
    html_content = template.replace('<!-- NEWS_LIST -->', '\n'.join(news_items))
    
    # 设置页面标题
    title_map = {'zh': '钢材新闻', 'en': 'Steel News', 'fr': 'Actualités de l\'acier', 'de': 'Stahl Nachrichten'}
    title_text = title_map.get(lang, 'Steel News')
    html_content = re.sub(r'<title>.*?</title>', f'<title>{title_text} - 巨红贸易</title>', html_content)
    return html_content

def upload_via_http(local_path, remote_filename):
    """通过 HTTP POST 上传文件内容到 PHP 脚本"""
    if not UPLOAD_TOKEN or not UPLOAD_URL:
        print("未设置 UPLOAD_TOKEN 或 UPLOAD_URL，跳过上传")
        return
    try:
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()
        payload = {
            'filename': remote_filename,
            'content': content
        }
        headers = {
            'X-Upload-Token': UPLOAD_TOKEN,
            'Content-Type': 'application/json'
        }
        response = requests.post(UPLOAD_URL, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            print(f"已上传 {local_path} -> {remote_filename}")
        else:
            print(f"上传失败，HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"HTTP 上传异常: {e}")

def main():
    print(f"{datetime.now()} - 开始抓取新闻...")
    new_news = fetch_rss_feeds()
    print(f"抓取到 {len(new_news)} 条新新闻")
    
    existing_news = load_existing_news()
    merged_news = merge_news(existing_news, new_news)
    print(f"合并后共 {len(merged_news)} 条新闻")
    save_news(merged_news)
    
    # 生成中文版
    zh_html = generate_html(merged_news, lang='zh')
    with open(OUTPUT_FILES['zh'], 'w', encoding='utf-8') as f:
        f.write(zh_html)
    print(f"已生成中文版：{OUTPUT_FILES['zh']}")
    upload_via_http(OUTPUT_FILES['zh'], OUTPUT_FILES['zh'])
    
    # 多语言
    if TRANSLATION_ENABLED:
        for lang in TRANSLATE_TO:
            print(f"正在翻译为 {lang}...")
            translated_news = [translate_news(n, lang) for n in merged_news]
            lang_html = generate_html(translated_news, lang=lang)
            with open(OUTPUT_FILES[lang], 'w', encoding='utf-8') as f:
                f.write(lang_html)
            print(f"已生成 {lang} 版：{OUTPUT_FILES[lang]}")
            upload_via_http(OUTPUT_FILES[lang], OUTPUT_FILES[lang])

if __name__ == '__main__':
    main()