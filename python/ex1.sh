#!/bin/bash 

screen -AdmS ex1_val -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#

screen -S ex1_val -X screen -t tab1 bash -lic "python e2st_exp.py --test 'Approx' >text/approx.txt"

screen -S ex1_val -X screen -t tab2 bash -lic "python e2st_exp.py --test 'exact' >text/exact.txt"
screen -S ex1_val -X screen -t tab3 bash -lic "python e2st_exp.py --test 'ER' >text/ER.txt"
screen -S ex1_val -X screen -t tab4 bash -lic "python e2st_exp.py --test 'ResampleB100' >text/ResampleB100.txt"
screen -S ex1_val -X screen -t tab5 bash -lic "python e2st_exp.py --test 'Cumulative' >text/Cumulative.txt"
screen -S ex1_val -X screen -t tab6 bash -lic "python e2st_exp.py --test 'CumulativeB100' >text/CumulativeB100.txt"

screen -S ex1_val -X screen -t tab7 bash -lic "python e2st_exp.py --test 'BiEdge' >text/BiEdge.txt"
screen -S ex1_val -X screen -t tab8 bash -lic "python e2st_exp.py --test 'BiEdgeB100' >text/BiEdgeB100.txt"

