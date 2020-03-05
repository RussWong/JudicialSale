
FROM python:3.7

WORKDIR /PAP

ADD . /PAP

RUN pip install -r requirements.txt

EXPOSE 9050

CMD ["/bin/bash"]