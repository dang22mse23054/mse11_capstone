FROM python:3.8.5
 
WORKDIR /app_py
 
COPY ./app_py/requirements.txt /app_py/
RUN pip install -r ./requirements.txt
 
EXPOSE 5000
 
# ENTRYPOINT python ./app.py
CMD tail -f /dev/null
