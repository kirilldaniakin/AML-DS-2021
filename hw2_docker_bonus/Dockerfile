FROM python:latest

ADD recommender.py /recommender/

WORKDIR /recommender/

COPY R.pt R.pt

COPY map_u.npy map_u.npy

COPY map_m.npy map_m.npy

COPY mf.pt mf.pt

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "python", "recommender.py" ]