FROM python:3.11.6-slim-bullseye
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
COPY . /app/
ENTRYPOINT ["/usr/local/bin/python", "run.py"]
