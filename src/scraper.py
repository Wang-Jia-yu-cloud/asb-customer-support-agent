import requests
import time
import re
from bs4 import BeautifulSoup

BASE_URL = "https://www.asb.co.nz"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-NZ,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

ALL_TAGS = [
    "Internet-Banking", "fastnet-classic", "Online-Applications", "Fees",
    "Foreign-Exchange", "Transfers", "International-Money-Transfer",
    "Automatic-Payment", "how-to", "Payments", "accounts", "Netcode",
    "Security", "two-step-verification", "asb-virtual", "investments",
    "kiwisaver", "Administration", "Limits", "fastnet-business",
    "clever-kash", "getwise", "teaching-kids", "Mobile-Banking",
    "ASB-Mobile-app", "General-Information", "recurring-payment",
    "contactless", "google-pay", "Password", "online-share-trading",
    "ASB-Login", "Bank-Accounts", "home-loans", "Interest-rates",
    "true-rewards", "cards", "atm", "Foreign-Currency", "eftpos",
    "asb-insurance", "ASB-Securities", "Apple-Pay", "contactless-payment",
    "Credit-Cards", "home-central", "Home-Loan", "personal-loan",
    "saving", "shares", "superannuation", "Tax",
]


def get_total_pages(tag: str) -> int:
    url = f"{BASE_URL}/help/tag.{tag}.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return 0
    soup = BeautifulSoup(resp.text, "html.parser")
    script = soup.find("script", string=lambda s: s and "totaltagResult" in s)
    if not script:
        return 1
    match = re.search(r'totaltagResult="(\d+)"', script.string)
    if not match:
        return 1
    total = int(match.group(1))
    return (total + 9) // 10


def get_links_from_index(tag: str) -> list:
    url = f"{BASE_URL}/help/tag.{tag}.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select("ul.articleList a[href^='/help/']"):
        href = a["href"].split("?")[0].split("#")[0]
        if not href.startswith("/help/tag.") and href not in links:
            links.append(href)
    return links


def get_links_from_xml(tag: str, page: int) -> list:
    url = f"{BASE_URL}/help/getArticles.{tag}.{page}.xml"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "xml")
    return [a.text.strip() for a in soup.find_all("pageurl")]


def collect_all_links() -> list:
    all_links = []
    for tag in ALL_TAGS:
        print(f"Processing tag: {tag}")
        tag_links = get_links_from_index(tag)
        total_pages = get_total_pages(tag)
        for page in range(1, total_pages + 1):
            new_links = get_links_from_xml(tag, page)
            for link in new_links:
                if link and link not in tag_links:
                    tag_links.append(link)
            time.sleep(0.3)
        before = len(all_links)
        for link in tag_links:
            if link not in all_links:
                all_links.append(link)
        print(f"  {len(tag_links)} articles, {len(all_links) - before} new unique")
        time.sleep(0.5)
    return all_links


def scrape_article(path: str) -> dict:
    url = BASE_URL + path if path.startswith("/") else path
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.find("h1")
    question = title.get_text(strip=True) if title else "NOT FOUND"

    body = soup.find("div", class_="sh-content")
    answer = body.get_text(separator="\n", strip=True) if body else "NOT FOUND"

    return {"question": question, "answer": answer, "url": url}


def clean_answer(question: str, answer: str) -> str:
    if answer.startswith(question):
        answer = answer[len(question):].strip()
    answer = re.sub(r"Last Updated:.*?\n", "", answer).strip()
    cutoff_phrases = [
        "Did this answer your question?",
        "Share on Facebook",
        "Share on Twitter",
        "Related articles",
        "Related\n",
    ]
    for phrase in cutoff_phrases:
        idx = answer.find(phrase)
        if idx != -1:
            answer = answer[:idx].strip()
    return answer.strip()


def run_scraper(output_path: str = "data/asb_faq_full.json"):
    import json
    links = collect_all_links()
    links = list(dict.fromkeys(links))
    print(f"\nTotal unique links: {len(links)}")

    results = []
    failed = []
    for i, path in enumerate(links):
        try:
            article = scrape_article(path)
            article["answer"] = clean_answer(article["question"], article["answer"])
            results.append(article)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(links)}] scraped so far...")
        except Exception as e:
            failed.append({"path": path, "error": str(e)})
        time.sleep(0.8)

    print(f"Done. Success: {len(results)}, Failed: {len(failed)}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")
