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

if [[ "$(docker images|cut -d' ' -f1|grep objectDetect)" != "objectDetect" ]]; then
    docker image load  -i ${WD}/data/od.repo.tar
fi

if [[ "$(docker ps -a|awk '{print $2}'|grep objectDetect)" == "objectDetect" ]]; then
    docker-compose start
else
    docker-compose up -d
fi