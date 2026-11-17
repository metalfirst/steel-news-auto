#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
钢铁新闻抓取 / Steel news feed builder
=======================================
在原有脚本基础上做了三处改动（其余逻辑保持不变）：

1. **把 Google News 跳转链接解析成发布方原始链接**，写入每条新闻的
   ``source_url`` 字段。Google News 的 ``/rss/articles/CBMi...`` 链接在
   国内无法访问；解析后前端可以直接跳到搜狐、中国钢铁新闻网等原始页面。
   解析失败时 ``source_url`` 为 ``null``，前端会自动退化为百度/必应搜索，
   不会出现死链。
2. 增加 ``NOISE_KEYWORDS``，过滤“钢铁侠/漫威”一类与钢铁行业无关的标题。
3. 修复 RSS 条目没有 ``published`` 字段时会抛 ``KeyError`` 的问题。

运行方式不变（GitHub Actions 里的 ``python update_news.py`` 会直接使用），
依赖也不变：requests / feedparser / scikit-learn / python-dateutil。
"""

import feedparser
import requests
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import quote, urlparse
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

# 需要屏蔽的关键词（包含即丢弃）
BLOCKED_KEYWORDS = ['钢铁森林']

# 只在标题里过滤的噪声词：Google News 的关键词搜索会把影视、体育内容带进来
NOISE_KEYWORDS = [
    '钢铁侠', '钢铁蜘蛛侠', '漫威', '复仇者联盟', '托尼斯塔克', '小罗伯特唐尼',
    '钢铁雄心', '蜘蛛侠',
    # “钢铁”的比喻用法：钢铁巨龙 / 钢铁之艺 / 钢铁意志 等，与行业无关
    '钢铁巨龙', '钢铁之艺', '钢铁意志', '钢铁洪流', '钢铁是怎样炼成的',
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

# ================= Google News 链接解析配置 =================
GOOGLE_NEWS_HOST = 'news.google.com'
GOOGLE_NEWS_RSS_ARTICLE = 'https://news.google.com/rss/articles/{article_id}'
BATCHEXECUTE_URL = 'https://news.google.com/_/DotsSplashUi/data/batchexecute'
GOOGLE_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
)
# 每次最多解析多少条（= 前端展示条数），避免无谓请求
MAX_RESOLVE = 50
# 解析签名时的间隔，防止被 Google 限流
RESOLVE_INTERVAL = 0.1
# garturlreq 请求里固定的上下文（来自 Google News 前端）
GARTURLREQ_CONTEXT = [
    ['X', 'X', ['X', 'X'], None, None, 1, 1, 'US:en', None, 1,
     None, None, None, None, None, 0, 1],
    'X', 'X', 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0,
]


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
    except Exception:
        try:
            # 兼容 RFC 822 格式
            return datetime.strptime(published_str, '%a, %d %b %Y %H:%M:%S %Z')
        except Exception:
            return None


def get_sort_key(news):
    """返回用于排序的 key，datetime 对象或原始字符串（作为后备）"""
    dt = parse_published(news.get('published', ''))
    return dt if dt else news.get('published', '')


def filter_blocked_news(news_list):
    """移除任何标题或内容中包含屏蔽关键词的新闻"""
    filtered = []
    for news in news_list:
        title = news.get('title', '') or ''
        content = news.get('content', '') or ''
        text_to_check = (title + ' ' + content).lower()
        # 如果任意屏蔽词出现在文本中，则跳过这条新闻
        if any(kw.lower() in text_to_check for kw in BLOCKED_KEYWORDS):
            continue
        # 影视/体育噪声只在标题里判断，避免误伤正文
        if any(kw.lower() in title.lower() for kw in NOISE_KEYWORDS):
            continue
        filtered.append(news)
    return filtered


# ================= Google News 原始链接解析 =================
def google_article_id(url):
    """从 Google News 链接里取出文章 id，非 Google 链接返回 None"""
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.hostname != GOOGLE_NEWS_HOST:
        return None
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) >= 2 and parts[-2] in ('articles', 'read'):
        return parts[-1]
    return None


def fetch_decode_params(article_id, session):
    """取一篇文章的 (timestamp, signature)。

    ``/articles/{id}`` 对非浏览器客户端会 302 到 sorry 页面；
    ``/rss/articles/{id}`` 则正常返回带 data-n-a-sg / data-n-a-ts 的 HTML。
    """
    url = GOOGLE_NEWS_RSS_ARTICLE.format(article_id=article_id)
    url += '?hl=en-US&gl=US&ceid=US:en'
    response = session.get(url, timeout=20, allow_redirects=True)
    response.raise_for_status()
    html = response.text
    signature = re.search(r'data-n-a-sg="([^"]+)"', html)
    timestamp = re.search(r'data-n-a-ts="([^"]+)"', html)
    if signature and timestamp:
        return timestamp.group(1), signature.group(1)
    return None


def decode_batch(pending, session):
    """一次 POST 解出多条链接。

    ``pending`` 是 ``(index, article_id, timestamp, signature)`` 列表；
    返回 ``[(request_id, url), ...]``。
    """
    if not pending:
        return []

    envelopes = []
    for request_id, article_id, timestamp, signature in pending:
        inner = [
            'garturlreq',
            GARTURLREQ_CONTEXT,
            article_id,
            int(timestamp) if str(timestamp).isdigit() else timestamp,
            signature,
        ]
        envelopes.append([
            'Fbv4je',
            json.dumps(inner, separators=(',', ':')),
            None,
            str(request_id),
        ])

    body = 'f.req=' + quote(json.dumps([envelopes], separators=(',', ':')))
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
        'User-Agent': GOOGLE_USER_AGENT,
        'Accept-Language': 'en-US,en;q=0.9',
    }
    response = session.post(BATCHEXECUTE_URL, data=body.encode('utf-8'),
                            headers=headers, timeout=30)
    response.raise_for_status()
    return parse_batchexecute(response.text)


def parse_batchexecute(text):
    """从 batchexecute 的响应里提取 (request_id, publisher_url)"""
    body = text
    if '\n\n' in body:
        body = body.split('\n\n', 1)[1]
    body = body.lstrip()
    if body.startswith(")]}'"):
        body = body.split('\n', 1)[1] if '\n' in body else body[4:]
        body = body.lstrip()

    rows = json.loads(body)
    if rows and isinstance(rows[-1], list) and rows[-1] and rows[-1][0] in ('di', 'e'):
        rows = rows[:-1]

    pairs = []
    for row in rows:
        if not (isinstance(row, list) and len(row) >= 3):
            continue
        payload = row[2]
        if row[0] == 'wrb.fr' or row[1] == 'Fbv4je':
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    continue
            if isinstance(payload, list) and payload and payload[0] == 'garturlres':
                request_id = None
                for cell in reversed(row[3:]):
                    if cell is not None:
                        request_id = str(cell)
                        break
                publisher_url = payload[1] if len(payload) > 1 else None
                if publisher_url:
                    pairs.append((request_id, publisher_url))
    return pairs


def resolve_source_urls(news_list, session):
    """为前 MAX_RESOLVE 条 Google News 新闻补上 ``source_url`` 字段。

    已经解析过、或者链接本来就不是 Google 的条目会被跳过，
    因此重复运行不会浪费请求。
    """
    targets = []
    for index, item in enumerate(news_list[:MAX_RESOLVE]):
        if item.get('source_url'):
            continue
        article_id = google_article_id(item.get('link', ''))
        if article_id:
            targets.append((index, article_id))

    if not targets:
        return 0

    pending = []
    for index, article_id in targets:
        try:
            params = fetch_decode_params(article_id, session)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                print('  Google 返回 429（限流），停止解析剩余链接')
                break
            print(f'  解析签名失败（{status}）: {article_id[:16]}…')
            params = None
        except Exception as exc:
            print(f'  解析签名失败: {article_id[:16]}… ({exc})')
            params = None

        if params:
            pending.append((index, article_id, params[0], params[1]))
        time.sleep(RESOLVE_INTERVAL)

    if not pending:
        return 0

    try:
        pairs = decode_batch(pending, session)
    except Exception as exc:
        print(f'  batchexecute 解析失败: {exc}')
        return 0

    by_id = {str(request_id): url for request_id, url in pairs if request_id is not None}
    ordered = [url for _, url in pairs]

    resolved = 0
    for offset, item in enumerate(pending):
        index = item[0]
        url = by_id.get(str(index))
        if url is None and not by_id and offset < len(ordered):
            url = ordered[offset]
        if url:
            news_list[index]['source_url'] = url
            resolved += 1
    return resolved


# ================= 核心函数 =================
def fetch_rss_feeds():
    all_news = []
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get('title', '')
                raw_content = entry.get('description', entry.get('summary', ''))
                clean_content = clean_html(raw_content)
                if not clean_content:
                    clean_content = title

                if any(kw.lower() in (title + clean_content).lower() for kw in KEYWORDS):
                    img_match = re.search(r'<img.*?src="(.*?)"', raw_content)
                    image_url = img_match.group(1) if img_match else None
                    link = entry.get('link', '')
                    news = {
                        'title': title,
                        'link': link,
                        'published': entry.get('published') or entry.get('updated') or '',
                        'source': feed.feed.get('title', '未知') if hasattr(feed, 'feed') else '未知',
                        'content': clean_content[:300] + ('...' if len(clean_content) > 300 else ''),
                        'image': image_url,
                        'id': hashlib.md5(link.encode()).hexdigest(),
                        'source_url': None,
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
            # 发布方原始链接；解析失败时为 None，前端会退化为站内可访问的搜索链接
            'source_url': news.get('source_url') or None,
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

    # 屏蔽包含屏蔽关键词的新闻
    merged_news = filter_blocked_news(merged_news)
    print(f"屏蔽后剩余 {len(merged_news)} 条新闻")

    # 把 Google News 跳转链接解析成发布方原始链接（国内可直接打开）
    if any(google_article_id(n.get('link', '')) for n in merged_news[:MAX_RESOLVE]):
        print("正在解析 Google News 原始链接...")
        with requests.Session() as session:
            session.headers.update({
                'User-Agent': GOOGLE_USER_AGENT,
                'Accept-Language': 'en-US,en;q=0.9',
            })
            # 减少出现 consent 中间页的概率
            session.cookies.set('CONSENT', 'YES+', domain='.google.com')
            resolved = resolve_source_urls(merged_news, session)
        print(f"已解析 {resolved} 条原始链接")

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
