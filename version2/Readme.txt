1. Data is not available here because the size of it. (if you want to replicate the process, just use the srcipts provided by sslime to download stl10 dataset and put in the root directory)
2. All log information are put in logs directory
3. all saved models are in checkpoints
4. The directory of "scripts" contains all the script that we used to run model.
5. The directory of "datasets" contains all the function that used to transform image


Reload module (we saved model as state dict):
	1. import specific model from directory "models", we used "alex_net.AlexNet" for SSL pretraining and "fusion_net.FusionNet"
	   for final classification problem
	2. follow the steps in any file whose name is start with "cls".
	3. choose which ssl model you want choose in the script.
	4. then run it.
