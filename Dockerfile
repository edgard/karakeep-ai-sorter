FROM python:3.14-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY karakeep_ai_sorter.py ./

# Default command; configure via env vars (see script docstring)
ENTRYPOINT ["python", "/app/karakeep_ai_sorter.py"]
