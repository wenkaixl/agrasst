#!/bin/bash 

screen -AdmS ex1_val -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#

screen -S ex1_val -X screen -t tab1 bash -lic "python e2st_exp.py --test 'Deg' >text/Deg.txt"

screen -S ex1_val -X screen -t tab2 bash -lic "python e2st_exp.py --test 'TV' >text/TV.txt"
screen -S ex1_val -X screen -t tab3 bash -lic "python e2st_exp.py --test 'Param' >text/Param.txt"
screen -S ex1_val -X screen -t tab4 bash -lic "python e2st_exp.py --test 'MDdeg' >text/MD.txt"

