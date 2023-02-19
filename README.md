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
The chosen model was ResNet50, which is a type of CNN architecture which is capable of classifying images.
For this architecture, the parameters tuned are:

- learning rate: the size of the steps that the algorithm will make when moving toward the minimum of the loss function.
- batch size: number of images that will be used for training at the same time, a correct batch size can reduce the training time.
- epochs: number of cycles that the algorithm will be trained on.

For every hyperparameter, the computer will select between two options, making a total of 8 possible combinations from which the computer will select the best combination of hyperparameters. This limited pool was selected to consume less computation resources and time (one train and test cycle consumed 40 minutes), however there are two possible approaches that can be used alone or in combination to expand the hyperparameter list of values, while not consuming extra time or resources:
1. Reduce training and test samples to less than 20% of the original dataset.
2. Use GPUs to train and test the model.

For this project

image of hyperparameter tuning job


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling

setting up rules and configuraations that i wanted to test, and that were aapplicable for the model training

modify the script to accept the hook aand its configuration from the parameters given by the SageMaker training job

initialize the hook within the script and pass it to the training and testing cycles of the model

**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
    
completed training jobs
logs during training

### Results

results: training aprox 8000 images for 3 train-and-then-test cycles took 4017 seconds around 1 hour and 6 minutes.

CPU utilization: 65% max
Memory utilization: 28% max, 27% most of the time

Computing resources were underutilized, a smaller machine could have done the same job, or the number of paarallel jobs increased in the saame machine

Also, none of the configured rules was violaated, but could have set up the rules that alerted me of the lower bound of resources


**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.

For the complete report, refer to the generated file in this folder `profiler-report.html`

## Model Deployment

query the endpint:
1. transform the image to a 4-dimensionaal PyTorch tensor, with this structure:
(batch_size, channels, width, height)
the batch size will always be 1, since the prediction runs on a single image, this is included because the pretrained model asks for it. to comply with the pretrained model specifications
2. resize the image into a 244 * 244 height, width
3. serialize the image
4. attaach it to the body of an http call, send it to the endpoint URL




**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions

For extensions to this project, it is recommended to:
- Package the trained model as a Docker image as explained [here](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
- Train and deploy different models using [Multi-Model Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)
- Deploy an endpoint that is capable of consuming many images for inference using [Batch Trnsforms](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- Use [Amazon Sagemaker Clarity](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html) to mke the model more interpretable
