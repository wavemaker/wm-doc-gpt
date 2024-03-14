FROM python:3.11.6-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app/
#CMD ["bash", "run.sh"]
ENTRYPOINT ["/usr/local/bin/python" , "/app/main.py"]