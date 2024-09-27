#!/bin/sh

# Parsing all arguments
# WIP : Manage exceptions and defaults
for i in "$@"
do
case $i in
    -o=*|--outdir=*)
    OUTDIR="${i#*=}"
    shift
    ;;
    -r=*|--repeat=*)
    REPEAT="${i#*=}"
    shift
    ;;
    -l=*|--layer=*)
    LAYER="${i#*=}"
    shift
    ;;
    -t=*|--targetTask=*)
    TASK="${i#*=}"
    shift
    ;;
    --seed=*)
    SEED="${i#*=}"
    shift
    ;;
    -h|--help)
    echo "Usage: ./start_expe -o|--outdir=outputdir -r|--repeat=30 -l|--layer=5 -t|--targetTask=cifar10 -s|--seed=0"
	exit 1
    shift
    ;;
    *)
    ;;
esac
done

# Echoing arguments to recap the user what he has typed
echo 'Script parameters : '
echo '\t Output directory: ' ${OUTDIR}
echo '\t Number of repeat: ' ${REPEAT}
echo '\t Layer: ' ${LAYER}
echo '\t Target task: ' ${TASK}
echo '\t Seed : ' ${SEED} '\n'

echo 'Starting run(s) : '

if [ "${REPEAT}" == "-1" ]; then
	echo "\n\t Layer " ${LAYER} " - Run number 1 \n"
	python3 quanta.py ${OUTDIR} 0 ${LAYER} ${TASK} ${SEED}
	exit 0
fi

total_run=0
for run in $(seq 1 $REPEAT)
do
	echo "\n\t Layer " ${LAYER} " - Run number " $run "\n"
	((total_run++))
	python3 quanta.py ${OUTDIR} $run ${LAYER} ${TASK} ${SEED}
done

echo "++++++++ REPEAT BASH SCRIPT DONE --- Layer ${LAYER} -- Total runs $total_run"
