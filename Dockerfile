FROM ufoym/deepo:pytorch-py36-cu111

RUN apt-get -y update
ADD requirements.txt /root/requirements.txt

RUN pip install --upgrade setuptools pip
RUN pip install -r /root/requirements.txt