# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir streamlit pandas scikit-learn matplotlib seaborn

# Expose port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
