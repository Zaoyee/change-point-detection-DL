# change-point-detection-DL


The code is already tested in the monsoon conda environment.
I waited for a GPU for half hour, but still did not get that.
Anyway, the code also works if there is no GPU, but it takes too much time.

You can run test_run.py script simply type
`python test_run.py`

The data is in the path of './data/processed/detailed/'

'datamat.csv' is the detailed dataset

'datamat2.csv' is the sysmetic dataset

In the `test_run.py` file, it should clarify the model and bunch of paramters before use the function.
and the the `test_run.py` will output the final prediction of one of testFold based on the number you inputs.
The results also include the loss and accuracy if needed.

`test_run.py` runs across all the validation fold, the based on the average of accuracy get an
optimal traning epoches; then use that to train all data except the testFold.

