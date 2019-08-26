#!/bin/bash
while true
do 
    procnum=`ps -ef|grep aaa.py|grep -v 'grep'|wc -l`
    echo $procnum
    if [ $procnum -eq 0 ];then  
        python aaa.py
    fi
    sleep 1
done

