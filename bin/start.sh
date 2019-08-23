#!/usr/bin/env bash

WD=$(cd $(dirname $(dirname $0)); pwd)

which yum-config-manager &> /dev/null
if [[ $? != 0 ]]; then
  yum install -y yum-utils
fi

if [[ ! -e /etc/yum.repos.d/docker-ce.repo ]]; then
  yum-config-manager --add-repo=${WD}/docker-ce.repo
fi

which docker &> /dev/null
if [[ $? != 0 ]]; then
  yum install -y docker-ce
  systemctl start docker
  systemctl enable docker
fi

which docker-compose &> /dev/null
if [[ $? != 0 ]]; then
  compose_version='1.24.0'
  curl -L https://github.com/docker/compose/releases/download/${compose_version}/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
  chmod +x /usr/local/bin/docker-compose
  curl -L https://raw.githubusercontent.com/docker/compose/master/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose
fi

if [[ "$(docker images|cut -d' ' -f1|grep object_detect)" != "object_detect" ]]; then
    docker image load  -i ${WD}/data/od.repo.tar
fi

if [[ "$(docker ps -a|awk '{print $2}'|grep object_detect)" == "object_detect" ]]; then
    docker-compose start
else
    docker-compose up -d
fi