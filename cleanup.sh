#!/bin/bash
yum -y clean all && rm -rf /var/cache/yum
rm -rf /etc/yum.repos.d/Artifactory.repo
rm -rf /etc/yum.repos.d/ArtifactoryDev.repo
ls -lrth /
