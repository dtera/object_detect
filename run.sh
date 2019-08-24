#!/bin/bash

yum install libSM libXext -y

nohup python3 server.py &> log/od.log &
