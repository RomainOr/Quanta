
#echo $0 # script name
#echo $1 # layer
#echo $2 # outputdir

# USAGE
# start_expe.sh layerNumber outputDir

layer=$1
output_dir=$2
total_run=0

echo $layer

for run in {0..9}
do
		echo "Layer " $layer " Run number " $run
		((total_run++))  
		python3 transferabilityFactor.py $layer $run $output_dir
done

for run in {10..19}
do
		echo "Layer " $layer " Run number " $run
		((total_run++))  
		python3 transferabilityFactor.py $layer $run $output_dir
done

for run in {20..29}
do
		echo "Layer " $layer " Run number " $run
		((total_run++))  
		python3 transferabilityFactor.py $layer $run $output_dir
done

echo " ++++++++ REPEAT BASH SCRIPT DONE --- Layer $layer   Total runs $total_run"
