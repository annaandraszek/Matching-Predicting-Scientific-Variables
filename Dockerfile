FROM python:3.5
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords
EXPOSE 5000
CMD python ./app.py