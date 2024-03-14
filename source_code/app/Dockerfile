FROM amazonlinux:2.0.20210126.0

ENV APP_ROOT /opt/app
ARG env
ARG git_branch
RUN \
  yum clean metadata; \
  yum update -y; \
  yum -y groupinstall "Development Tools"; 
RUN yum clean all;
RUN yum -y install net-tools httpd initscripts procps\
	which zip wget tar perl libaio libaio-devel passwd git\
	openssh openssh-server openssh-clients sudo vim python-setuptools openssl mod_ssl;
RUN yum clean all;
RUN yum install python3.7 -y
RUN pip3 install aws-encryption-sdk-cli
# Setting ssh
RUN sed -i 's/#Port 22/Port 22/g' /etc/ssh/sshd_config
RUN cd /etc/ssh/ && ssh-keygen -A
RUN mkdir ~/.ssh
RUN echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCYo90rrVHxb67dLSO01A3oWH1pP6ewQrFoKzdaFIfxiUeK8yDTh/1S6+fHtMkfT8FqmVesV5FApdn69d18k+hXb6BUxjDTLeHfLBVg0pNSOWzsj0RhwyF0Y5lmJ74QK8WnXFwvwWldgqz4Cl4B+QUJrqD2nPYSqG5wFbryRaZY+cs5WhnoaUX7pLL6p6jF5FsQJ/PXTjpWvD1q/aFWeW3r4zehnbqpmiLZScpLqQBTdHV+3joGvxvZxMxzor8so18DYQumrbRFrGOJIGlPBk39nIFLBYvqLGhiUy7JIBxkk0aZ32+pKP87JZ387pYL1PTRfIM2QIqhj3Ws/Dnx8Wb5 task-tracker" > ~/.ssh/authorized_keys
RUN chmod 600 ~/.ssh/authorized_keys

# Setting git
# Replace gitkey path of your PC
COPY ./git_key /root/.ssh/git_key
RUN chmod 400 ~/.ssh/git_key
RUN printf "Host github.com\nIdentityFile ~/.ssh/git_key\nStrictHostKeyChecking no\nUserKnownHostsFile /dev/null" >> ~/.ssh/config
RUN cd ~/ && git clone git@github.com:CyberAgent-Infosys/caad-task-tracker.git
RUN cd ~/caad-task-tracker/ && git fetch origin && git checkout $git_branch && git pull
# Autodeploy
RUN printf "ENV=${env}" >> /root/.bash_profile
COPY ./autodeploy.sh /root/autodeploy.sh

# #install nodejs
RUN curl -sL https://rpm.nodesource.com/setup_15.x | sudo bash -
RUN yum -y install nodejs
RUN npm install npm@latest -g
RUN npm install --global yarn
RUN npm install knex -g
RUN npm install -g typescript
RUN npm install -g ts-node
RUN npm install yarn -g
RUN npm install pm2@latest -g

# Install credentials
RUN mkdir /root/.aws
COPY ./credentials /root/.aws/credentials

# Clone source
RUN mkdir $APP_ROOT

COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["sh", "./entrypoint.sh"]
EXPOSE 3000
EXPOSE 22