"""Command line: uv run ask "why did payments fail?" (the service must be running)."""

import argparse
import httpx
def main() -> None:
    parser = argparse.ArgumentParser(description="Ask why something failed; answered from runbooks.")
    parser.add_argument("question")
    parser.add_argument("--url", default="http://localhost:8002", help="where the service is running")
    parser.add_argument("--max-chunks", type=int, default=5)
    args = parser.parse_args()

    response = httpx.post(f"{args.url}/ask", json={"question": args.question, "max_chunks": args.max_chunks},
                          timeout=300)
    response.raise_for_status()
    data = response.json()

    print(data["answer"])
    if data["sources"]:
        print("\nSources:")
        for source in data["sources"]:
            print(f"  [{source['n']}] {source['source_uri']}")


if __name__ == "__main__":
    main()
