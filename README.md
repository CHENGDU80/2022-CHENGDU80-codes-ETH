# chengdu80-codes
## How to run our code
To install the environment, use the following command:
```
conda env create -f environment.yml
```
You should have the folder named `model` and `submission` in this directory, the folder should be organized like this:
```
|-code
|   |-model
|   |-submission
|   |-model.py
|   |-other files in github
|-data
|   |-train
|       |-feature.csv
|       |-label.csv
|   |-test
|       |-feature.csv
|       |-label.csv
|   |-final
|       |-feature.csv
|       |-sample.csv
```
To train our final model, run:
```
python model.py --stage train --model_name gb_brf_2 
```
Now you will get a file named `model_gb_brf_2 + timestamp.pkl` in the model folder. 

To evaluate our final model on the given test set, run:
```
python model.py --stage evaluate --modelfile_name model_gb_brf_21667933542
```
The modelfile_name is the model you have in the model directory. Feel free to change it to any model you have in the `model` directory.

To generate the final submission dataset, run:
```
python model.py --stage test --modelfile_name model_gb_brf_21667933542
```

Then you will find a new submit csv file in the submission directory.


Same as above you can play around with the `model_name` in the training stage with proper string values given in our `model.py`, e.g. brf_brf.


For some other models, in `model_avg.py` we use the average combination of some clustered features. You can also run the code similarly as in  `model.py`, e.g.
```
python model_avg.py --stage train --model_name avg_gb_brf_2
```
Since the outcome is not good, we don't provide the test stage to generate the submission csv.


In `data_exp.ipynb`, we provide some of our data exploration thoughts, and generating data analytic figures for our final presentation.  

In `figures.ipynb`, we provide some evaluation figures of our final model. The timestamp of some of the model may be changed, but you can replace it with your newly generated timestamp and it will function the same.