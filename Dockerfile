FROM python:3.12-slim

# Install poppler for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY static/ ./static/

# Create upload directory
RUN mkdir -p uploads

# Server config
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Set your enterprise OpenAI key here (or pass at runtime)
# ENV OPENAI_API_KEY=sk-...

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
