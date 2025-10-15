# Use official Python image as base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#working directory
WORKDIR /app

COPY requirements.txt ./
#dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#executable bit
RUN chmod +x /app/mcp-servers/mcp-postgres


# Streamlit needs to know it’s inside a container
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false


EXPOSE 8501
# Launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
