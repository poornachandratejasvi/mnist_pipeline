#!/bin/bash
yum install -y python3 elfutils-libs procps shadow-utils vim-minimal wget python3-devel openssl bind-utils tar
yum install -y gcc
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip --version
python3 -m pip install --no-cache-dir -r /bd_build/requirements.txt
chmod 777 -R /tmp/
