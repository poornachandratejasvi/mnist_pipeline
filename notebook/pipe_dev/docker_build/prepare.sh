#!/bin/bash
dest="/opt/workflow/"
mkdir -p $dest
cp -r "/bd_build/etl/" "/opt/workflow/"
cp -r "/bd_build/coordinatorService/" "/opt/workflow/"
cp -r "/bd_build/intermediate/topology/" "/opt/workflow/"
cp -r "/bd_build/common" "/opt/workflow/"
cp -r "/bd_build/alarm_log_Sendclient" "/opt/workflow/"
cp "/bd_build/Artifactory_NSW.repo" "/etc/yum.repos.d/"
cp "bd_build/ca.sh" "/tmp/"
cp -r "/bd_build/ETL.sh" "/opt/workflow/"
cp -r "/bd_build/COS.sh" "/opt/workflow/"
ls /opt/workflow/
yum install -y python3 elfutils-libs procps shadow-utils vim-minimal wget python3-devel openssl bind-utils
yum install -y gcc
yum install -y --enablerepo="Artifactory_NSW" caclients
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip --version
python3 -m pip install --no-cache-dir -r /bd_build/requirements.txt

mkdir /certs/
chmod 777 /certs
