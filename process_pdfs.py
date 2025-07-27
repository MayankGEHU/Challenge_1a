import fitz
import json
import re
import unicodedata
from pathlib import Path
from statistics import median
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import joblib
from difflib import SequenceMatcher

# ---------------------- Utilities ----------------------
def clean_title(text):
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    result, last = [], None
    for w in words:
        if w != last:
            result.append(w)
        last = w
    cleaned = ' '.join(result)
    cleaned = re.sub(r'(\b\w+\b)(?:\s+\1\b)+', r'\1', cleaned)
    return cleaned.strip()

normalize_text = lambda t: re.sub(r'\W+', '', t).lower()
unicode_normalize = lambda t: unicodedata.normalize('NFKD', t)

def fuzzy_match(a, b, threshold=0.9):
    return SequenceMatcher(None, a, b).ratio() >= threshold

# ---------------------- Block Feature Extraction ----------------------
def get_block_features(block):
    lines = block.get('lines', [])
    if not lines:
        return None
    raw = ' '.join(''.join(span['text'] for span in line['spans']) for line in lines).strip()
    if not raw:
        return None
    text = unicode_normalize(raw)
    sizes = [span['size'] for line in lines for span in line['spans']]
    font_size = median(sizes) if sizes else 0
    font_name_flags = [span.get('font', '').lower() for line in lines for span in line['spans']]
    is_bold = any('bold' in f or 'black' in f or 'heavy' in f for f in font_name_flags)
    pct_upper = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    hp = re.compile(r'^\s*(\d+(\.\d+)*\.?|Appendix [A-Z]|[A-Z]\.|\u2022|-|\u2013)\s+')
    starts_pattern = bool(hp.match(text))
    return {
        'text': clean_title(text),
        'font_size': font_size,
        'is_bold': is_bold,
        'line_count': len(lines),
        'line_length': len(text),
        'pct_uppercase': pct_upper,
        'starts_pattern': starts_pattern,
        'normalized': normalize_text(text)
    }

def extract_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page in doc:
        raw = page.get_text('dict')['blocks']
        for b in raw:
            if b.get('type') != 0:
                continue
            fb = get_block_features(b)
            if fb:
                fb['page'] = page.number + 1
                all_blocks.append(fb)
    return all_blocks, len(doc)

# ---------------------- Vectorization ----------------------
def vectorize(blocks):
    X = []
    for i, b in enumerate(blocks):
        prev = blocks[i - 1] if i > 0 else {k: 0 for k in b}
        nxt = blocks[i + 1] if i < len(blocks) - 1 else {k: 0 for k in b}
        vec = [
            b['font_size'], int(b['is_bold']), b['line_count'], b['line_length'], b['pct_uppercase'],
            int(b['starts_pattern']),
            prev.get('font_size', 0), int(prev.get('is_bold', False)), prev.get('line_count', 0),
            prev.get('line_length', 0), prev.get('pct_uppercase', 0), int(prev.get('starts_pattern', False)),
            nxt.get('font_size', 0), int(nxt.get('is_bold', False)), nxt.get('line_count', 0),
            nxt.get('line_length', 0), nxt.get('pct_uppercase', 0), int(nxt.get('starts_pattern', False))
        ]
        X.append(vec)
    return X

# ---------------------- Title Extraction ----------------------
def extract_largest_font_title(blocks, tolerance=0.5):
    page1_blocks = [b for b in blocks if b['page'] == 1 and len(b['text'].strip()) > 5]
    if not page1_blocks:
        return ""
    scored_blocks = sorted(
        page1_blocks,
        key=lambda b: (-b['font_size'], -int(b['is_bold']), -b['pct_uppercase'], -len(b['text']))
    )
    top_font = scored_blocks[0]['font_size']
    title_lines = [b['text'].strip() for b in scored_blocks if abs(b['font_size'] - top_font) <= tolerance]
    seen, result = set(), []
    for line in title_lines:
        if line and line not in seen:
            result.append(line)
            seen.add(line)
    return " ".join(result).strip()

def patch_title_for_special_cases(title, outline, blocks):
    if not title.strip():
        if outline:
            return outline[0]['text']
        return extract_largest_font_title(blocks)
    return title

# ---------------------- Output Generation ----------------------
def generate_output(input_dir, output_dir, model_path):
    clf = joblib.load(model_path)
    for pdf in sorted(input_dir.glob('*.pdf')):
        blocks, pages = extract_blocks(pdf)
        if not blocks:
            continue
        X = vectorize(blocks)
        preds = clf.predict(X)
        title = ''
        outline = []
        heading_labels = [lbl for lbl in set(preds) if re.match(r'^H\d+$', lbl)]
        for b, label in zip(blocks, preds):
            if label == 'TITLE' and not title:
                title = b['text']
            elif label in heading_labels:
                outline.append({'level': label, 'text': b['text'], 'page': b['page'] - 1})

        if not title:
            title = extract_largest_font_title(blocks)
        patched_title = patch_title_for_special_cases(title, outline, blocks)

        # ------------------- ToC Removal Logic (Generic) -------------------
        for h in outline:
            h['text'] = re.sub(r'\s+\d{1,3}$', '', h['text'].strip())
            h['norm'] = normalize_text(h['text'])

        last_occurrence = {}
        for h in outline:
            if h['norm'] not in last_occurrence or h['page'] > last_occurrence[h['norm']]:
                last_occurrence[h['norm']] = h['page']

        final_outline = []
        seen = set()
        for h in outline:
            key = (h['level'], h['norm'], h['page'])
            if key in seen:
                continue
            if h['page'] == last_occurrence[h['norm']]:
                final_outline.append({k: h[k] for k in ['level', 'text', 'page']})
                seen.add(key)

        final_outline.sort(key=lambda h: (h['page'], h['text']))
        data = {
            'title': clean_title(patched_title),
            'outline': final_outline
        }
        outp = output_dir / f"{pdf.stem}.json"
        outp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding='utf-8')
        print(f"Wrote {outp}: {len(final_outline)} headings")

# ---------------------- Label Alignment ----------------------
def align_labels(blocks, ground_truth):
    gt_map = {normalize_text(o['text']): o['level'] for o in ground_truth.get('outline', [])}
    title_norm = normalize_text(ground_truth.get('title', ''))
    labels = []
    for b in blocks:
        norm = b['normalized']
        if norm == title_norm or fuzzy_match(norm, title_norm):
            labels.append('TITLE')
        elif norm in gt_map or any(fuzzy_match(norm, k) for k in gt_map):
            labels.append(gt_map.get(norm) or next(gt_map[k] for k in gt_map if fuzzy_match(norm, k)))
        else:
            labels.append('BODY')
    return labels

# ---------------------- Training ----------------------
def train_model(pdf_dir, json_dir, model_path):
    X_all, y_all = [], []
    for pdf in pdf_dir.glob('*.pdf'):
        blocks, pages = extract_blocks(pdf)
        jpath = json_dir / f"{pdf.stem}.json"
        if not jpath.exists():
            continue
        gt = json.loads(jpath.read_text(encoding='utf-8'))
        y = align_labels(blocks, gt)
        X = vectorize(blocks)
        X_all.extend(X)
        y_all.extend(y)
    if not X_all:
        print("No training data.")
        return
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    clf.fit(X_all, y_all)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"RF model saved at {model_path}")

# ---------------------- Main ----------------------
if __name__ == '__main__':
    data_pdf = Path('data/pdfs')
    data_json = Path('data/jsons')
    model_file = Path('models/rf_block_model.pkl')
    inp = Path('input')
    out = Path('output')

    data_pdf.mkdir(exist_ok=True, parents=True)
    data_json.mkdir(exist_ok=True, parents=True)
    inp.mkdir(exist_ok=True, parents=True)
    out.mkdir(exist_ok=True, parents=True)

    if not model_file.exists():
        train_model(data_pdf, data_json, model_file)
    else:
        print(f"Model exists: {model_file}, skipping training.")

    generate_output(inp, out, model_file)
    print("Done.")