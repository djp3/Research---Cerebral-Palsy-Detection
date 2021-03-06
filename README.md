# Research---Cerebral-Palsy-Detection
Using OpenCV, AWS ML, and AWS Sagemaker we attempt to predict signs of cerebral palsy in premature babies


## To Run the Dense_Optical_Flow.py file:

Install Anaconda [For Windows](https://docs.anaconda.com/anaconda/install/windows/),
                 [For Linux](https://docs.anaconda.com/anaconda/install/linux/),
                 [For Mac](https://docs.anaconda.com/anaconda/install/mac-os/)

### Create a virtual environment:

1. Open an ancaconda prompt
		
2. Type `conda create -n yourenvironmentname`
		
3. Activate your new env with `source activate yourenvironmentname`

[Creating Virtual Environments with Anaconda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

*Note: You can also create a new environment using the GUI provided in the Anaconda Navigator*

*Note: You will also need to install packages such as OpenCV and pymysql to compile the Dense_Optical_Flow program*

### Install required packages:
1. Open an anaconda prompt

2. Navigate to your environment that you are using, if it is seperate from the base environment

3. Type `conda install opencv`

4. Type `conda install pymysql`


### Final Steps

Install spyder using the Anaconda Navigator. I used Spyder2.3.8 but the latest version should be fine

Use spyder to open, run, or edit the Dense_Optical_Flow.py program

### Turnover stuff

Things to give Prof. Patterson

* Database access:
	* Done. Database is on AWS under Patterson's root credentials
* Accelerometer
	* Ability to convert raw accelerometer in the database to generated in the db Python script that runs outside of AWS to capture the raw accelerometer data and put the generated data into AWS.
		* Done. The python script is in `2019_Summer_Workflow/01_Accelerometer_Raw_To_Generated/feature_extraction.ipynb`
	* Ability to train an XGBoost model on accelerometer generated data and test it 
		* Done.  The jupyter notebook is in SageMaker called nested_kfold_xgb_from_db.ipynb: https://xl.notebook.us-east-1.sagemaker.aws/notebooks/nested_kfold_xgb_from_db.ipynb
* Video	
	* Ability to convert raw rgb frames in the database to optical flow images in the db.  A Python script that runs outside of AWS to capture the raw accelerometer data and put the generated data into AWS.
		* Done. The python script is in `2019_Summer_Workflow/02_Video_Raw_To_Generated/dense_optical_flow.py`
	* Ability to convert depth frames in the database into depth-flow images
		* Not done.  First task for summer students


		
