import json
import sys
import urllib.request

BASE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=test"
)
PAGE = 100


def main(out_path):
    texts = []
    offset = 0
    total = None
    while total is None or offset < total:
        url = f"{BASE}&offset={offset}&length={PAGE}"
        with urllib.request.urlopen(url) as r:
            payload = json.load(r)
        if total is None:
            total = payload["num_rows_total"]
            print(f"{total} rows total")
        rows = payload["rows"]
        if not rows:
            break
        texts.extend(row["row"]["text"] for row in rows)
        offset += len(rows)
        print(f"\r{offset}/{total}", end="", flush=True)

    body = "".join(texts)
    with open(out_path, "w") as f:
        f.write(body)
    print(f"\nwrote {len(body)} chars ({len(texts)} rows) to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/wikitext2-raw-test.txt")
