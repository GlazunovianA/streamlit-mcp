# Stage 1: Build the MCP PostgreSQL server
FROM node:22.12-alpine AS builder

# Copy the base tsconfig that the postgres config extends
COPY servers-archived/tsconfig.json /mcp/tsconfig.json

# Copy the postgres server directory
COPY servers-archived/src/postgres /mcp/src/postgres

WORKDIR /mcp/src/postgres

# Install dependencies and build
RUN --mount=type=cache,target=/root/.npm npm install
RUN --mount=type=cache,target=/root/.npm npm run build

# Stage 2: Create a minimal Node runtime with built server
FROM node:22-alpine AS release

COPY --from=builder /mcp/src/postgres/dist /app/dist
COPY --from=builder /mcp/src/postgres/package*.json /app/

WORKDIR /app

ENV NODE_ENV=production
RUN npm ci --ignore-scripts --omit=dev

# Stage 3: Build the Streamlit application
FROM python:3.11-slim

# Install Node.js runtime for executing the MCP server
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy Python application files
COPY requirements.txt ./
COPY app.py ./
COPY toolbox.py ./
COPY .streamlit ./.streamlit

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180
RUN pip install --upgrade pip && \
    pip install --retries 5 --timeout 180 -r requirements.txt
    
# Copy the built MCP server from release stage
RUN mkdir -p /app/mcp-servers/mcp-postgres
COPY --from=release /app/dist /app/mcp-servers/mcp-postgres/dist
COPY --from=release /app/package*.json /app/mcp-servers/mcp-postgres/

# Install Node dependencies for the MCP server
WORKDIR /app/mcp-servers/mcp-postgres
RUN npm ci --omit=dev --ignore-scripts && \
    chmod +x dist/index.js && \
    node --version && \
    ls -la dist/

# Return to app directory
WORKDIR /app

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

EXPOSE 8501

# Launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]