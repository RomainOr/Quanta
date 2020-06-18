import sys
import os
import math as m
import tensorflow as tf
import numpy as np
from tf_models import createModel
from tf_models import learning_rates
from tf_models import get_vinit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from binaryChoiceActivation import binaryChoiceActivation

#tf.compat.v1.disable_eager_execution()

# Run 1st run for gradual evaluation with layer 5:
# ./start_expe.sh -o outputDir --co_eval=false --current_run=0 --layer=5
# Run 30rd run for co-evaluation:
# ./start_expe.sh -o outputDir --co_eval=true --current_run=29
outputDir  = sys.argv[1]
currentRun = int(sys.argv[3])
trsf_layer = None
if sys.argv[2] == 'false':
	trsf_layer = int(sys.argv[4])
	tf_coeval  = False
else: tf_coeval = True

trainSource = False
targetData  = 'cifar100'

#Parametres
eta, etaTF = learning_rates()
etaDecay	 	  = 1e-6	#lr decay for optimizer
numberOfEpochsSource  = 60
numberOfEpochsWitness = 60
numberOfEpochsTarget  = 60
batchSizeSource		  = 32	
batchSizeTarget		  = 32	

optimizer	= tf.keras.optimizers.Adam(eta, decay=etaDecay)	#Optimizer for gradient descent
loss		= 'categorical_crossentropy'	#Loss giving gradients
#accuracy	= ['accuracy']					#Metric for accuracy
metrics     = ['accuracy', \
				tf.keras.metrics.Precision(),\
				tf.keras.metrics.Recall(),   \
				tf.keras.metrics.FalsePositives(), \
				tf.keras.metrics.FalseNegatives(), \
				tf.keras.metrics.TruePositives(),  \
				tf.keras.metrics.TrueNegatives(),  \
				tf.keras.metrics.TopKCategoricalAccuracy(k=3)]

augmentData 		= True	 #Flow  data augmentation during training
fromPreviousTraining= False	 #False for training from scratch, True to start from previous save

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	#Less verbosity



######## INPUT DATA
print("Loading Data")
(trainingSetSource, trainingLabelsSource),(testSetSource, testLabelsSource) = \
		tf.keras.datasets.cifar10.load_data()
if targetData == 'cifar100':
	(trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
		tf.keras.datasets.cifar100.load_data()
else:
	(trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
		tf.keras.datasets.cifar10.load_data()

print("Normalizing data, creating one-hots")
## Normalize data and create one hots
trainingSetSource    = trainingSetSource/255.
trainingLabelsSource = tf.keras.utils.to_categorical(trainingLabelsSource, 10)
testSetSource    = testSetSource/255.
testLabelsSource = tf.keras.utils.to_categorical(testLabelsSource, 10)

trainingSetTarget = trainingSetTarget/255.
if targetData == 'cifar100':
	trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, 100)
else:
	trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, 10)

testSetTarget = testSetTarget/255.
if targetData == 'cifar100':
	testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, 100)
else:
	testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, 10)

inputPlaceholderSource = tf.keras.Input([32, 32, 3], name='inputHolderSource')
inputPlaceholderTarget = tf.keras.Input([32, 32, 3], name='inputHolderTarget')


"""
"""
## Data augmentation tensor
trainGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
	featurewise_center=False,  		#set input mean to 0 over the dataset
    samplewise_center=False,  		#set each sample mean to 0
    featurewise_std_normalization=False,  	#divide inputs by std of the dataset
    samplewise_std_normalization=False,  	#divide each input by its std
    zca_whitening=False,  			#apply ZCA whitening
    zca_epsilon=1e-06,  			#epsilon for ZCA whitening
    rotation_range=45,  			#randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,			#randomly shifting image horizontally
    height_shift_range=0.1,			#randomly shifting image vertically
    shear_range=0.1,  			    #set range for random shear
    zoom_range=0.1,  				#set range for random zoom
    channel_shift_range=0.,  		#set range for random channel shifts
    fill_mode='nearest',			#fillmode for image manipulation
    cval=0.,  						#value used for fill_mode = "constant"
    horizontal_flip=True,  			#randomly flip images horizontally
    vertical_flip=True,  			#randomly flip images vertially
    rescale=None,
    preprocessing_function=None,
    data_format=None
)

######## BUILDING MODELS
NNTarget  = None
NNSource  = None
NNWitness = None
NNSourceCopy = None

def buildModels(NNSource, NNSourceCopy, NNTarget, NNWitness):
	tf_type = 'coeval'
	if tf_coeval is False: tf_type = 'gradual'

	NNSource     = tf.keras.Model(inputs=inputPlaceholderSource,
					outputs=createModel(inputPlaceholderSource, 10, trsf_type=tf_type, trsf_layer=trsf_layer)) 
	NNSourceCopy = tf.keras.Model(inputs=inputPlaceholderTarget,
				outputs=createModel(inputPlaceholderTarget, 10, trsf_type=tf_type, trsf_layer=trsf_layer)) 

	if targetData == 'cifar100':
		NNTarget  = tf.keras.Model(inputs=inputPlaceholderTarget, 
						outputs=createModel(inputPlaceholderTarget, 100,
								sourceModel=NNSourceCopy, trsf_type=tf_type, trsf_layer=trsf_layer)) 
		NNWitness = tf.keras.Model(inputs=inputPlaceholderTarget,
				outputs=createModel(inputPlaceholderTarget, 100, trsf_type=tf_type, trsf_layer=trsf_layer))
	else:
		NNTarget  = tf.keras.Model(inputs=inputPlaceholderTarget,
						outputs=createModel(inputPlaceholderTarget, 10,
								sourceModel=NNSourceCopy, trsf_type=tf_type, trsf_layer=trsf_layer)) 
		NNWitness = tf.keras.Model(inputs=inputPlaceholderTarget,
				outputs=createModel(inputPlaceholderTarget, 10, trsf_type=tf_type, trsf_layer=trsf_layer))
	
	for l in NNSourceCopy.layers: l.trainable=False
	NNSource.compile(optimizer, loss, metrics)
	NNTarget.compile(optimizer, loss, metrics)
	NNWitness.compile(optimizer, loss, metrics)
	
	# [Anne] ???
	NotLoadSource  = True
	NotLoadTarget  = True
	NotLoadWitness = True
	return ((NNSource, NNSourceCopy, NNTarget, NNWitness),(NotLoadSource, NotLoadTarget, NotLoadWitness))

print("Building source and target models")
(NNSource, NNSourceCopy, NNTarget, NNWitness),(NotLoadSource, NotLoadTarget, NotLoadWitness) = \
			buildModels(NNSource, NNSourceCopy, NNTarget, NNWitness)


### DEBUG
#NNSourceCopy.summary(line_length=100)
#NNTarget.summary(line_length=100)
#sys.exit()
####################

nbrLambdas = len([NNTarget.layers[i].get_weights() \
					for i in range(len(NNTarget.layers)) \
					if (type(NNTarget.layers[i]).__name__ == "binaryChoiceActivation")])

############
### TF Keras callbacks for retrieving lambdas
lambdas = [[] for i in range(0, nbrLambdas)]
raw_lambdas = [[] for i in range(0, nbrLambdas)]

def _sigmoid(x): return 1./(1+np.exp(-x))

def lambdasMonitoring(w): 
	weights = [np.array(w[i])[0][0] for i in range(len(w))]
	for i in range(0, nbrLambdas):
		lambdas[i] += [_sigmoid(weights[i][0])]
		raw_lambdas[i] += [weights[i][0]]
	return

def lambdasMonitoring_VERSION2(w): 
	weights = [np.array(w[i])[0][0] for i in range(len(w))]
	for i in range(0, nbrLambdas):
		lambdas[i] += [m.exp(weights[i][0])/(m.exp(weights[i][0])+m.exp(weights[i][1]))]
		raw_lambdas[i] += [weights[i][0]]
	return

scalarWeightCallback = tf.keras.callbacks.LambdaCallback(
		on_epoch_end=lambda epoch,logs: lambdasMonitoring(    #_VERSION2(
				[NNTarget.layers[i].get_weights()           \
				for i in range(len(NNTarget.layers))        \
				if (type(NNTarget.layers[i]).__name__ == "binaryChoiceActivation")]))

def lambdas_to_str(lambdas):
	s = ""
	for l in range(nbrLambdas): s += str(round(lambdas[l][-1],4)) +' '
	return s

printLambdas = tf.keras.callbacks.LambdaCallback(
		on_epoch_end=lambda epoch,logs: \
			print("\nTF : " + lambdas_to_str(lambdas) ) ) \

########
# Helper callbacks (for debugging and or understanding)
printRawLambdas = tf.keras.callbacks.LambdaCallback(
		on_epoch_end=lambda epoch,logs: \
			print('Raw lambdas: ' + lambdas_to_str(raw_lambdas), ' '))

def print_BCA_layers(model):
	s= ""
	for l in range (len(model.layers)):
		if (type(model.layers[l]).__name__ == "binaryChoiceActivation"):
			s += 'Layer ' + str(l) + ' ' + str(model.layers[l].get_weights()) + '\n'
	return s

print_raw_BCA_weights = tf.keras.callbacks.LambdaCallback(
		on_epoch_begin=lambda epoch,logs: \
			print('BCA parameters:\n', print_BCA_layers(NNTarget)))

#########
# Callback for exporting accuracy DURING training ?
target_metrics = []
record_target_metrics = tf.keras.callbacks.LambdaCallback(
	on_epoch_end=lambda epoch,logs: target_metrics.append(
						NNTarget.evaluate(testSetTarget, testLabelsTarget)))

###### MODEL TRAINING AND EVALUATION
def train(modelName, dataAugmentation=False, fromPreviousTraining=False): 
	#modelName : T for target, S for source, W for witness
	if (modelName == 'T'):
		print("Training target model...")
		NNSourceCopy.load_weights('trained_models/NNSource_w.h5')
		model = NNTarget
		trainingSet    = trainingSetTarget
		trainingLabels = trainingLabelsTarget
		weights = "trained_models/NNTarget_w.h5"
		noe = numberOfEpochsTarget
		cb  = [scalarWeightCallback, printLambdas, printRawLambdas, \
			   record_target_metrics] #, print_raw_BCA_weights]
		bc  = batchSizeTarget
		NotLoadTarget=False

	elif (modelName == 'S'):
		print("Training source model...")
		model = NNSource
		trainingSet    = trainingSetSource
		trainingLabels = trainingLabelsSource
		weights = "trained_models/NNSource_w.h5"
		cb  = []
		noe = numberOfEpochsSource
		bc  = batchSizeSource
		NotLoadSource=False
	else :
		print("Training witness model...")
		model = NNWitness
		trainingSet    = trainingSetTarget
		trainingLabels = trainingLabelsTarget
		weights = "trained_models/NNWitness_w.h5"
		cb  = []
		noe = numberOfEpochsWitness
		bc  = batchSizeSource
		NotLoadWitness=False

	if dataAugmentation: 
		trainGenerator.fit(trainingSet, augment=True)
		if fromPreviousTraining:model.load_weights(weights)
		model.fit_generator(
			trainGenerator.flow(
				trainingSet, trainingLabels, batch_size=bc), epochs=noe, callbacks=cb)
	else: 
		if fromPreviousTraining: model.load_weights(weights)
		NNSource.fit(trainingSet, trainingLabels, epochs=noe, callbacks=cb, batch_size=bc)
	
	print("Saving model parameters")
	model.save_weights(weights)
	return

def test(modelName): 
	if (modelName == 'T'):
		print("Testing target model...")
		NNSourceCopy.load_weights("trained_models/NNSource_w.h5")
		model = NNTarget
		testSet    = testSetTarget
		testLabels = testLabelsTarget
		weights = "trained_models/NNTarget_w.h5"
		snl     = NotLoadTarget

	elif (modelName == 'S'):
		print("Testing source model...")
		model = NNSource
		testSet    = testSetSource
		testLabels = testLabelsSource
		weights = "trained_models/NNSource_w.h5"
		snl     = NotLoadWitness
	else :
		print("Testing witness model...")
		model = NNWitness
		testSet    = testSetTarget
		testLabels = testLabelsTarget
		weights = "trained_models/NNWitness_w.h5"
		snl     = NotLoadSource

	if snl: model.load_weights(weights)

	metrics = model.evaluate(testSet, testLabels)

	f = open(outputDir+'/metrics'+str(currentRun)+str(modelName)+'.txt', 'w')
	f.write(str(metrics)+'\n')
	f.close()
	print("Final testing accuracy :" +str(metrics[1])+"\n")
	return metrics[1] # accuracy

def resetVariables(model): #resets all weigths layer by layers in the model
	# /!\ NB doesnt seem to reset the model !
	session = tf.keras.backend.get_session()
	for l in model.layers: 
		for v in l.__dict__:
			vArg = getattr(l,v)
			if hasattr(vArg, 'initializer'):
				initializer = getattr(vArg, 'initializer')
				initializer.run(session=session)
	(NNSource, NNSourceCopy, NNTarget, NNWitness),(NotLoadSource, NotLoadTarget, NotLoadWitness) = \
			buildModels(NNSource, NNSourceCopy, NNTarget, NNWitness)
	return

def writeFactors(l, e, raw=False):
	if raw is True: f = open(outputDir+'/'+str(e)+'_raw.txt', 'w')
	else:           f = open(outputDir+'/'+str(e)+'.txt', 'w')
	out = ""
	for i in range(0, nbrLambdas):
		for j in range(0, len(l[i])):
			out += str(l[i][j])+" " 
		out += ","
	f.write(out)
	f.close()

def export_expe_summary(NNTarget, tf_type, target_task, src_accuracy, target_accuracy):
	f = open(outputDir+'/expe_summary.txt', 'a')
	export  = 'Target task: ' + targetData + '\n'
	export += 'LRM: ' + str(etaTF/eta) + '(Eta: '+str(eta)+' EtaTF: '+str(etaTF)+ ')\n'
	export += 'v init: ' + str(get_vinit()) + '\n'
	export += 'Transfer type: ' + str(tf_type) + '\n' + \
			 str(target_task) + '\nSource model accuracy :' + \
			 str(float(src_accuracy)) + \
			 '\nTarget model accuracy: ' + str(target_accuracy) + \
			 '\nTarget model summary:\n'
	f.write(export)
	NNTarget.summary(line_length=80, print_fn=lambda x: f.write(x + '\n'))
	f.close()
	return

def main1():
	#with tf.compat.v1.Session() as s:

		#s.run(tf.compat.v1.global_variables_initializer())
		#s.run(tf.compat.v1.local_variables_initializer())
	
		#NNTarget.summary()

		#train('S', augmentData, fromPreviousTraining)
		#test('S') #0.8358

		#train('W', augmentData, fromPreviousTraining)
		#test('W') #0.5628

	print("Target task: ", targetData)

	print('Creating output directory')
	if not os.path.exists(outputDir): os.makedirs(outputDir)	

	print('Starting run ', currentRun)

	for j in range(0, nbrLambdas): lambdas[j] = []
	train('T', augmentData, fromPreviousTraining)
	writeFactors(lambdas, currentRun, raw=False)
	writeFactors(raw_lambdas, currentRun, raw=True)
	if currentRun == 0:
		export_expe_summary(NNTarget, 'co-eval' if tf_coeval else 'gradual',
							targetData, test('S'), test('T'))
		
	f = open(outputDir+'/all_target_metrics_' + str(currentRun) + '.txt','a')
	for i in range(0, len(target_metrics)): f.write(str(target_metrics[i])+'\n')
	f.close()
	#print("cleaning up...\n")
	#resetVariables(NNTarget)
	#plotLambdaTraining()
	return

if __name__=="__main__":
	main1()
