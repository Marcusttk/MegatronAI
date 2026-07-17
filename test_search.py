from search import SearchEngine

engine = SearchEngine()

results = engine.search(
    "running",
    k=20
)

for i, r in enumerate(results, start=1):

    print("=" * 60)

    print(f"Rank {i}")

    print(f"Similarity : {r['score']:.3f}")

    print(f"Author     : {r['author']}")

    print(f"Channel    : {r['channel_id']}")

    print()

    print(r["content"])

    print()