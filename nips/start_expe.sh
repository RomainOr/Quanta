

# USAGE
# Usage: 
# ./start_expe -o|--outdir=outputdir -r|--currentRun=29 -l|--layer=5" -t|--targetTask='cifar10'|'cifar100'

total_run=0

for i in "$@"
do
case $i in
    -o=*|--outdir=*)
    OUTDIR="${i#*=}"
    shift # past argument=value
    ;;
    -r=*|--currentRun=*)
    CURRENTRUN="${i#*=}"
    shift # past argument=value
    ;;
    -l=*|--layer=*)
    LAYER="${i#*=}"
    shift # past argument=value
    ;;
    -t=*|--targetTask=*)
    TASK="${i#*=}"
    shift # past argument=value
    ;;
    -h|--help)
    echo "Usage: ./start_expe -o|--outdir=outputdir -r|--currentRun=29 -l|--layer=5 -t|targetTask=cifar10"
	exit 1
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done
echo 'Output directory: ' ${OUTDIR}
echo 'Running simultaneous transfer: ' ${COEVAL}
echo 'Current run (out of 30): ' ${CURRENTRUN}
echo 'Layer: ' ${LAYER}
echo 'Target task: ' ${TASK}

echo 'Starting run...'

if [ "${CURRENTRUN}" == "-1" ]; then # only 1 run for testing
	echo "Layer " ${LAYER} " Run number " $run
	python3 quanta.py ${OUTDIR} 0 ${LAYER} ${TASK}
	exit 0
fi


for run in {0..29}
do
	echo "Layer " ${LAYER} " Run number " $run
	((total_run++))  
	python3 quanta.py ${OUTDIR} $run ${LAYER} ${TASK}
done


echo " ++++++++ REPEAT BASH SCRIPT DONE --- Layer ${LAYER} -- Total runs $total_run"
echo "     Target task: ${TASK}
