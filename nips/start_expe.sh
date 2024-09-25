

# USAGE
# Usage: 
# ./start_expe -o|--outdir=outputdir -r|--repeat=30 -l|--layer=5" -t|--targetTask='cifar10'|'cifar100'

total_run=0

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
    -h|--help)
    echo "Usage: ./start_expe -o|--outdir=outputdir -r|--repeat=30 -l|--layer=5 -t|targetTask=cifar10"
	exit 1
    shift
    ;;
    --default)
    DEFAULT=YES
    shift
    ;;
    *)
    ;;
esac
done
echo 'Echo shell parameters : '
echo '\t Output directory: ' ${OUTDIR}
echo '\t Running simultaneous transfer: ' ${COEVAL}
echo '\t Number of repeat: ' ${REPEAT}
echo '\t Layer: ' ${LAYER}
echo '\t Target task: ' ${TASK} '\n'

echo 'Starting run(s) : '

if [ "${REPEAT}" == "-1" ]; then
	echo "\n\t Layer " ${LAYER} " - Run number 1 \n"
	python3 quanta.py ${OUTDIR} 0 ${LAYER} ${TASK}
	exit 0
fi


for run in $(seq 1 $REPEAT)
do
	echo "\n\t Layer " ${LAYER} " - Run number " $run "\n"
	((total_run++))
	python3 quanta.py ${OUTDIR} $run ${LAYER} ${TASK}
done

echo "++++++++ REPEAT BASH SCRIPT DONE --- Layer ${LAYER} -- Total runs $total_run"


#TODO : valeurs par défaut et exceptions