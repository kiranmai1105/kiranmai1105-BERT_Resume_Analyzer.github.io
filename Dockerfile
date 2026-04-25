FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python src/train.py

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]