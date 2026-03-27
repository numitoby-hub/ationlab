"""
arXiv 논문 자동 수집 & 주제별 분류
─────────────────────────────────
실행: python fetch_papers.py
결과: archcad/papers/ 아래 주제별 폴더에 PDF + 메타데이터 저장
"""

import os, re, time, json, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "papers")
MAX_RESULTS_PER_QUERY = 50          # 쿼리당 최대 결과 수
DOWNLOAD_PDF = True                  # False면 메타데이터만 저장
MIN_DATE = "2024-01-01"

# ── 주제별 검색 쿼리 ──────────────────────────────────────────────
# key: 폴더명, value: arXiv 검색 쿼리
TOPIC_QUERIES = {
    "panoptic_segmentation": [
        "panoptic segmentation transformer",
        "mask2former panoptic",
        "panoptic segmentation query-based",
    ],
    "cad_floorplan": [
        "CAD drawing recognition",
        "floorplan recognition deep learning",
        "architectural drawing parsing",
        "building plan segmentation",
        "CAD primitive detection",
    ],
    "graph_neural_network": [
        "graph attention network node classification",
        "GATv2 graph neural network",
        "graph neural network segmentation",
    ],
    "vision_transformer": [
        "SegFormer semantic segmentation",
        "vision transformer dense prediction",
        "feature pyramid vision transformer",
    ],
    "set_prediction": [
        "hungarian matching object detection",
        "DETR-like set prediction",
        "bipartite matching instance segmentation",
    ],
}

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """arXiv API로 논문 검색 후 파싱된 리스트 반환."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    print(f"  검색: {query}")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read()
    except Exception as e:
        print(f"    ⚠ 요청 실패: {e}")
        return []

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall("atom:entry", NS):
        published = entry.find("atom:published", NS).text[:10]
        if published < MIN_DATE:
            continue

        title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)
        summary = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
        summary = re.sub(r"\s+", " ", summary)

        arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]
        # v1, v2 등 버전 제거
        arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

        pdf_link = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")
                break

        authors = [a.find("atom:name", NS).text
                    for a in entry.findall("atom:author", NS)]

        categories = []
        for c in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
            categories.append(c.get("term"))
        for c in entry.findall("atom:category", NS):
            t = c.get("term")
            if t and t not in categories:
                categories.append(t)

        papers.append({
            "arxiv_id": arxiv_id_clean,
            "title": title,
            "authors": authors,
            "published": published,
            "summary": summary,
            "pdf_url": pdf_link,
            "categories": categories,
        })

    return papers


def safe_filename(title: str, max_len: int = 80) -> str:
    """제목을 파일명으로 변환."""
    name = re.sub(r"[^\w\s-]", "", title)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:max_len]


def download_pdf(url: str, path: str):
    """PDF 다운로드."""
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"    ⚠ PDF 다운 실패: {e}")
        return False


def main():
    print("=" * 60)
    print("arXiv 논문 수집기 — PanCADNet v2 관련")
    print(f"기간: {MIN_DATE} ~ 현재")
    print("=" * 60)

    seen_ids = set()
    total_saved = 0

    for topic, queries in TOPIC_QUERIES.items():
        topic_dir = os.path.join(BASE_DIR, topic)
        os.makedirs(topic_dir, exist_ok=True)

        topic_papers = []
        print(f"\n📂 [{topic}]")

        for q in queries:
            results = search_arxiv(q)
            for paper in results:
                if paper["arxiv_id"] in seen_ids:
                    continue
                seen_ids.add(paper["arxiv_id"])
                topic_papers.append(paper)
            time.sleep(3)  # arXiv API rate limit 준수

        # 날짜순 정렬
        topic_papers.sort(key=lambda p: p["published"], reverse=True)

        for paper in topic_papers:
            fname = f"{paper['published']}_{safe_filename(paper['title'])}"
            paper_dir = os.path.join(topic_dir, fname)
            os.makedirs(paper_dir, exist_ok=True)

            # 메타데이터 저장
            meta_path = os.path.join(paper_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(paper, f, ensure_ascii=False, indent=2)

            # PDF 다운로드
            if DOWNLOAD_PDF and paper["pdf_url"]:
                pdf_path = os.path.join(paper_dir, f"{paper['arxiv_id']}.pdf")
                if not os.path.exists(pdf_path):
                    download_pdf(paper["pdf_url"], pdf_path)
                    time.sleep(1)

            total_saved += 1

        print(f"  → {len(topic_papers)}편 저장")

    # ── 전체 인덱스 생성 ──────────────────────────────────────────
    index_path = os.path.join(BASE_DIR, "index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"# arXiv 논문 수집 결과\n")
        f.write(f"수집일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"기간: {MIN_DATE} ~\n")
        f.write(f"총 {total_saved}편\n\n")

        for topic in TOPIC_QUERIES:
            topic_dir = os.path.join(BASE_DIR, topic)
            f.write(f"## {topic}\n\n")
            metas = []
            for root_d, dirs, files in os.walk(topic_dir):
                for file in files:
                    if file == "meta.json":
                        with open(os.path.join(root_d, file), encoding="utf-8") as mf:
                            metas.append(json.load(mf))
            metas.sort(key=lambda p: p["published"], reverse=True)
            for p in metas:
                authors_str = ", ".join(p["authors"][:3])
                if len(p["authors"]) > 3:
                    authors_str += " et al."
                f.write(f"- **{p['title']}**\n")
                f.write(f"  {authors_str} | {p['published']} | `{p['arxiv_id']}`\n\n")

    print(f"\n{'=' * 60}")
    print(f"완료! 총 {total_saved}편 → {BASE_DIR}")
    print(f"인덱스: {index_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
