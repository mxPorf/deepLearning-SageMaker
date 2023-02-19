# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
The chosen model was ResNet50, which is a type of CNN architecture capable of classifying images, as requested by the project directions.
For this model, the parameters tuned are:

- learning rate: the size of the steps that the algorithm will make when moving toward the minimum of the loss function.
- batch size: number of images that will be used for training at the same time, a correct batch size can reduce the training time.
- epochs: number of cycles that the algorithm will be trained on.

For every hyperparameter, the computer will select between two options, making a total of 8 possible combinations from which the computer will select the best combination of hyperparameters. 

Using this options, the optimal hyperparameters resulted as follows:
![best_hyperparameters.png](https://github.com/mxPorf/deepLearning-SageMaker/blob/44066838d226f3ef3f4013abb1202e88d3dbe6a6/images/best_hyperparameters.png)

The limited hyperprameter pool was selected to consume less computation resources and time (one train and test cycle consumed 40 minutes), however there are two possible approaches that can be used alone or in combination to expand the hyperparameter list of values, while not consuming extra time or resources:
1. Reduce training and test samples to less than 20% of the original dataset.
2. Use GPUs to train and test the model.

## Debugging and Profiling
Using Sagemaker hooks added to the training script, the program collected information about the training and test progression, and then compiled reports that can be used to optimize training times, increase model performance and identify malfunctions within the training and test cycle.
As stated, the training script (_train_model.py_) is modified to accept Sagemaker hooks, while an additional script (_train_and_deploy.ipynb_) is required to upload the desired hook configuration to Sagemaker and initiate the training job.
For data extraction, a single hook object can gather debug and profiling information, and it is instantiated from the training script, then the model is registered with the hook so that the hook can start gathering data when the model is being trained (hook set in train mode) and when the results are tested (hook set in test mode)

Additionaly, these screenshots document the completion of the trainig job as well as the progression of the loss metric during each epoch of the training:

![test_logs.png](https://github.com/mxPorf/deepLearning-SageMaker/blob/44066838d226f3ef3f4013abb1202e88d3dbe6a6/images/test_logs.png)
![completed_training_job.png](https://github.com/mxPorf/deepLearning-SageMaker/blob/44066838d226f3ef3f4013abb1202e88d3dbe6a6/images/completed_training_job.png)

### Results
The complete train and test cycle took 4017 seconds of billing time in AWS, which is aroung 1 hour and 6 minutes.
In this prooject, the hook was asked to collect data and to output alerts if the computing instances used for training were being overloaded. The profiling results, however, suggest that computing resources were underutilized in this case, the course of action can be to downsize the training instances used or to provision parallel jobs to train the model in less time, with the same provisioned resources. In future deployments, the user is encouraged to configure alerts to be aware of the underuse of resource, in addition to the configuration used here.

Some relevant statistics that point to the conclusion that the training instances were underutilized are the following: 
- CPU usage: 65% maximum, 50% on average
- Memory usage: 28% maximum, 27% on average

The reader is encouraged to consult the `profiler-report.html` file to look through the complete report and create their own conclusion.

## Model Deployment
Finally, the trained model was packaged with the script that contains instructions on how to run prodictions on the model, and uploaded to a provisioned EC2 instance with an endpoint that can be utilized to ask for inferences with custom data.

The process to query the input is as follows:
1. Transform the image to a 4-dimensional PyTorch tensor, with this structure: (batch_size, channels, width, height).
the batch size will always be 1, since the prediction runs on a single image, this is included to comply with the pretrained model specifications.
2. Resize the image to a 244 * 244 image.
3. Serialize the image data.
4. Attach it to the body of an http call, send it to the endpoint URL.
5. The endpoint responds with an array of probabilities.

These steps can be acomplished with PytTorch to make the necessary transformations and the Sagemaker.pytorch.Estimator object to serialize and send the request

Screenshot of the active endpoint, including the URL that services inference requests:
![active_endpoint](https://github.com/mxPorf/deepLearning-SageMaker/blob/46bb804c01fd96cef52709422b2fe8bdeb5aaf1f/images/active_endpoint.png?raw=true)

## Standout Suggestions

For extensions to this project, it is recommended to:
- Package the trained model as a Docker image as explained [here](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
- Train and deploy different models using [Multi-Model Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)
- Deploy an endpoint that is capable of consuming many images for inference using [Batch Trnsforms](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- Use [Amazon Sagemaker Clarity](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html) to mke the model more interpretable
