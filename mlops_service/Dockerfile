FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

# Shell form (not exec/JSON array) so $PORT actually expands - most PaaS
# hosts (Render included) assign a dynamic port via this env var and expect
# the app to bind to it. Falls back to 8000 for local/Docker Compose use,
# where PORT is never set.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
