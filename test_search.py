from search import SearchEngine


def main() -> None:
    with SearchEngine() as engine:
        results = engine.search(
            "people who enjoy running",
            k=20,
        )

        for rank, result in enumerate(results, start=1):
            print("=" * 70)
            print(f"Rank       : {rank}")
            print(f"Similarity : {result['score']:.3f}")
            print(f"Author     : {result['author']}")
            print(f"Channel    : {result['channel_name']}")
            print(f"Channel ID : {result['channel_id']}")
            print(f"Message ID : {result['id']}")
            print(f"Jump URL   : {result['jump_url']}")
            print()
            print(result["content"])
            print()


if __name__ == "__main__":
    main()