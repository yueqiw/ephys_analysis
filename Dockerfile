FROM python:3.6-jessie

WORKDIR /app

COPY requirements.txt /app 
RUN pip install -r /app/requirements.txt

COPY allensdk_0_14_2 /app/allensdk_0_14_2
COPY assets /app/assets
COPY visualization /app/visualization
COPY *.py /app/

RUN mkdir /app/data
VOLUME /app/data


COPY config.json /app
COPY run.sh /app

CMD /app/run.sh
