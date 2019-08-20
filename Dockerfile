# Use an official Python runtime as a parent image
FROM python:3.6-alpine

# Define environment variable
ENV FLASK_APP=re-kg FLASK_ENV=production

# Set the working directory to /usr/local/app/re
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . ./

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && mv wait-for /bin/ && \
    chmod +x /bin/wait-for && \
    rm -f requirements.txt
# RUN pip3 install -i http://pypi.douban.com/simple --trusted-host pypi.douban.com -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run server.py when the container launches
ENTRYPOINT ["python", "server.py"]