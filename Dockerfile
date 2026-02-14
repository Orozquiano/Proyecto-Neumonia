# Imagen base con Python 3.12
FROM python:3.12-slim

# Evita archivos .pyc y habilita logs inmediatos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar curl y dependencias básicas
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Agregar uv al PATH
ENV PATH="/root/.local/bin:$PATH"

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY pyproject.toml uv.lock ./

# Crear entorno virtual e instalar dependencias
RUN uv sync --frozen

# Copiar el resto del código
COPY . .

# Activar entorno virtual automáticamente
ENV PATH="/app/.venv/bin:$PATH"

# Comando por defecto
CMD ["python", "main.py"]

