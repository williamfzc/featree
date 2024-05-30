import ollama


class LLM(object):
    def ask(self, prompt: str) -> str:
        pass


class OllamaLLM(LLM):
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


class MockLLM(LLM):
    def ask(self, prompt: str) -> str:
        return ""


def get_llm() -> LLM:
    return OllamaLLM()


def get_mock_llm() -> LLM:
    return MockLLM()
