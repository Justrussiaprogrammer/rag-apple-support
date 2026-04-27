import httpx

from app.config import settings
from app.rag.retreiver import Retriever


SYSTEM_PROMPT = """
Ты — ассистент службы поддержки Apple.

Твоя задача — ответить на вопрос пользователя строго на основе переданного контекста.

Правила:
1. Используй только информацию из контекста.
2. Не добавляй сведения, которые не нужны для ответа на вопрос.
3. Если в контексте есть несколько похожих фрагментов, выбери самый прямой и релевантный.
4. Не пересказывай все найденные фрагменты подряд.
5. Если информации в контексте недостаточно, честно скажи: "В базе знаний нет достаточной информации для ответа."
6. Не выдумывай факты, ссылки, причины или инструкции.
7. Отвечай кратко, понятно и по делу.
8. Если вопрос предполагает инструкцию, дай ответ в виде коротких шагов.
9. Не упоминай номера фрагментов, если пользователь об этом не просит.
10. Не говори, что ты языковая модель.

Формат ответа:
1. Краткий ответ.
2. Затем шаги, если они нужны.
3. Не добавляй лишние разделы.
"""


def build_prompt(question, retrieved_chunks):
    context_blocks = list()
    for i, item in enumerate(retrieved_chunks):
        chunk = item.chunk
        context_blocks.append(
            f"""
            Фрагмент {i}
            Документ: {chunk.doc_title}
            Раздел: {chunk.section_title}
            Текст: {chunk.text}
            """
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""
            {SYSTEM_PROMPT}

            Контекст:
            {context}

            Вопрос пользователя:
            {question}

            Ответ:
        """


class OllamaGenerator:
    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
            },
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise ValueError(
                "Не удалось подключиться к Ollama. Проверьте, что Ollama запущена: "
                "<ollama serve> или системный сервис Ollama"
            )
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"Ollama вернула ошибку: {exc.response.text}") from exc

        data = response.json()
        answer = data.get("response", "").strip()
        if not answer:
            return "Модель не вернула ответ"
        return answer


class RAGService:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = OllamaGenerator()

    def retrieve(self, query, top_k):
        return self.retriever.retrieve(query, top_k or settings.retrieve_top_k)

    def ask(self, question):
        retrieved = self.retrieve(question, settings.retrieve_top_k)

        if not retrieved or retrieved[0].score < settings.retrieve_score_threshold:
            return "В базе знаний нет достаточной информации для ответа.", retrieved

        context_chunks = retrieved[: settings.max_context_chunks]
        prompt = build_prompt(question, context_chunks)
        answer = self.generator.generate(prompt)

        return answer, context_chunks
