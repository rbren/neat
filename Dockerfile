FROM python:3.7-slim

WORKDIR /neat
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD python /neat/lib/train.py
