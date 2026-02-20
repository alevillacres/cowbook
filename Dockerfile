# Usa un'immagine base ottimizzata per ML (es. Ultralytics o PyTorch)
FROM ultralytics/ultralytics:latest

# Imposta la directory di lavoro
WORKDIR /app

# Copia i requisiti e installali
COPY requirements.txt .
RUN pip install --no-cache-dir "fastapi[standard]" uvicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY . .

# Esponi la porta utilizzata da FastAPI (default 8000)
EXPOSE 8000

# Comando per avviare il server API
CMD ["python3", "-m", "fastapi", "run", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]