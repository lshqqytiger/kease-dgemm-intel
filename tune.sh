#!/bin/bash

set -e

export OMP_NUM_THREADS=68
export OMP_PLACES=cores
export OMP_PROC_BIND=close

OUT_PATH=./out
LOG_PATH=./log

function run(){
    rm -f ./out/dgemm_flops_kernel.01.mc.out
    make -s kernel.01.mc BLOCK="-DCM=$1 -DMB=$2 -DNB=$3 -DKB=$4"

    echo CM=$1 MB=$2 NB=$3 KB=$4 >> ${LOG_PATH}/log_cm$1
    ${OUT_PATH}/dgemm_flops_kernel.01.mc.out 68 Col N N 6000 6000 6000 -1.0 1.0 5 >> ${LOG_PATH}/log_cm$1
    gflops=$(tail -n3 ${LOG_PATH}/log_cm$1 | awk '/best/ {print $3*1000}')
    echo $gflops
}

for cm in 1 2 4 17 34 68
do
    echo "Tunning CM=$cm case..."

    b_mb=$((400 + ($RANDOM % 11 - 5) * 8))
    b_nb=$((48 + ($RANDOM % 3 - 1) * 24))
    b_kb=$((64 + ($RANDOM % 11 - 5) * 8))
    b_gflops=0

    for iter in {1..10..1}
    do
        echo "....iteration $iter"
        echo "....mb=$b_mb nb=$b_nb kb=$b_kb gflops=$b_gflops"

######################## mb ##################

        mb=$b_mb
        nb=$b_nb
        kb=$b_kb

        echo "....Tunning mb..."

        mb=$(($mb-16 > 8 ? $mb-16 : 8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        mb=$(($mb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        mb=$(($mb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        mb=$(($mb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        mb=$(($mb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        mb=$(($mb-56))
        if [ "$mb" -gt "7" ]
        then
            echo -n "........running mb=$mb nb=$nb kb=$kb case..."
            gflops=$(run $cm $mb $nb $kb)
            echo -n " gflops=$gflops"
            if [ "$gflops" -gt "$b_gflops" ]
            then
                b_mb=$mb
                b_nb=$nb
                b_kb=$kb
                b_gflops=$gflops
                echo " -- UPDATE!"
            else
                echo ""
            fi
        fi

        mb=$(($mb+80))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

######################## nb ##################

        mb=$b_mb
        nb=$b_nb
        kb=$b_kb

        echo "....Tunning nb..."

        nb=$(($nb-48 > 24 ? $nb-48 : 24))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        nb=$(($nb+24))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        nb=$(($nb+24))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        nb=$(($nb+24))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        nb=$(($nb+24))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        nb=$(($nb-168))
        if [ "$nb" -gt "23" ]
        then
            echo -n "........running mb=$mb nb=$nb kb=$kb case..."
            gflops=$(run $cm $mb $nb $kb)
            echo -n " gflops=$gflops"
            if [ "$gflops" -gt "$b_gflops" ]
            then
                b_mb=$mb
                b_nb=$nb
                b_kb=$kb
                b_gflops=$gflops
                echo " -- UPDATE!"
            else
                echo ""
            fi
        fi

        nb=$(($nb+240))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

######################## kb ##################

        mb=$b_mb
        nb=$b_nb
        kb=$b_kb

        echo "....Tunning kb..."

        kb=$(($kb-16 > 40 ? $kb-16 : 40))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        kb=$(($kb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        kb=$(($kb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        kb=$(($kb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        kb=$(($kb+8))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi

        kb=$(($kb-56))
        if [ "$kb" -gt "7" ]
        then
            echo -n "........running mb=$mb nb=$nb kb=$kb case..."
            gflops=$(run $cm $mb $nb $kb)
            echo -n " gflops=$gflops"
            if [ "$gflops" -gt "$b_gflops" ]
            then
                b_mb=$mb
                b_nb=$nb
                b_kb=$kb
                b_gflops=$gflops
                echo " -- UPDATE!"
            else
                echo ""
            fi
        fi

        kb=$(($kb+80))
        echo -n "........running mb=$mb nb=$nb kb=$kb case..."
        gflops=$(run $cm $mb $nb $kb)
        echo -n " gflops=$gflops"
        if [ "$gflops" -gt "$b_gflops" ]
        then
            b_mb=$mb
            b_nb=$nb
            b_kb=$kb
            b_gflops=$gflops
            echo " -- UPDATE!"
        else
            echo ""
        fi
    done
    echo "Tuned! [cm=$cm mb=$b_mb nb=$b_nb kb=$b_kb] gflops=$b_gflops"
done

echo "DONE!!"
