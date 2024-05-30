import ollama


class LLM(object):
    def ask(self, prompt: str) -> str:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.1,
            },
        )
        return response["message"]["content"]
