# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Expose port if needed (for API)
EXPOSE 8000

# Default command
CMD ["uv", "run", "python", "scripts/quick_test.py"]