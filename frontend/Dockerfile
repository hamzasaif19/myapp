FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir streamlit joblib pandas scikit-learn

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
