
#echo $0 # script name
#echo $1 # layer
#echo $2 # outputdir

# USAGE
# start_expe.sh layerNumber outputDir

total_run=0

for i in "$@"
do
case $i in
    -o=*|--outdir=*)
    OUTDIR="${i#*=}"
    shift # past argument=value
    ;;
    -c=*|--coeval=*)
    COEVAL="${i#*=}"
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
    -h|--help)
    echo "Usage: ./start_expe -o|--outdir=outputdir -c|--coeval=false -r|--currentRun=29 -l|--layer=5"
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
echo 'Layer (if gradual eval): ' ${LAYER}

if [ "${CURRENTRUN}" == "-1" ]; then
	echo "Layer " ${LAYER} " Run number " $run
	python3 tf.py ${OUTDIR} ${COEVAL} 0 ${LAYER}
	exit 0
fi

exit 42

for run in {0..29}
do
	echo "Layer " ${LAYER} " Run number " $run
	((total_run++))  
	python3 tf.py ${OUTDIR} ${COEVAL} $run ${LAYER}
done


echo " ++++++++ REPEAT BASH SCRIPT DONE --- Layer $layer   Total runs $total_run"
