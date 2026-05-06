# RAG Support Service

Простой RAG-сервис для ответов на вопросы по русскоязычным страницам Apple Support.

## Что делает сервис

Пайплайн:

```text
вопрос пользователя -> retrieval по FAISS -> контекст -> локальная LLM через Ollama -> ответ + sources
```

## Стек

- Python 3.12
- FastAPI
- FAISS
- `intfloat/multilingual-e5-small`
- Ollama c `qwen2.5:1.5b`

## Установка

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip git curl

python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Ollama

Установить Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Скачать модель:

```bash
ollama pull qwen2.5:1.5b
```

Проверить:

```bash
ollama run qwen2.5:1.5b
```

## Запуск пайплайна

### 1. Скачать документы Apple Support

```bash
python scripts/fetch_apple_docs.py
```

Результат:

```text
data/raw/apple_docs.jsonl
```

Если какие-то страницы не скачаются, можно заменить URL в `data/source_urls.json`.

### 2. Построить индекс

```bash
python scripts/build_index.py
```

Результат:

```text
data/index/faiss.index
data/index/chunks.jsonl
```

### 3. Запустить API

```bash
uvicorn app.main:app --reload
```

Swagger:

```text
http://localhost:8000/docs
```

### 4. Запуск через Docker

Систему можно запустить с помощью докера. Для этого уже должен быть построен индекс из пункта 2. Нужно 2 команды:

```bash
docker compose up --build
```

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```

После этого система будет работать самостоятельно

## Запросы

### Health

```bash
curl http://localhost:8000/health
```

### Retrieval

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "Как обновить iPhone?", "top_k": 3}'
```

### Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Как создать резервную копию iPhone?"}'
```

## Оценка retrieval

```bash
python scripts/evaluate.py
```

Метрика:

```text
Recall@3
```

Файл результата:

```text
eval/results.json
```

## Chunking

Используется секционный чанкинг:

1. Документ преобразуется в Markdown-подобный текст.
2. Текст делится по заголовкам `h1/h2/h3`.
3. Слишком длинные секции режутся на части до 1200 символов.
4. При принудительном разрезании используется overlap 150 символов.
