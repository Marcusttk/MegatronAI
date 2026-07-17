import ollama

stream = ollama.chat(
    model="gemma4:12b",
    messages=[
        {
            "role": "user",
            "content": "Say hello."
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk)