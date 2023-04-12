# Gradient Base Image

This repo houses a Dockerfile used to create an image used for Gradient runtimes. This image should be used for all Gradient runtimes, either solely or as the base layer for other images.
 
The goal of this image is to provide common packages for an advanced data science user that allows them to utilize the GPU hardware on Gradient. This image targets popular Machine Learning frameworks and includes packages commonly used for Computer Vision and Natural Language Processing tasks as well as general Data Science work.


## Software included

| Category         | Software         | Version                | Install Method | Why / Notes |
| -------------    | -------------    | -------------          | -------------  | ------------- |
| GPU              | NVidia Driver    | 510.73.05              | pre-installed  | Enable Nvidia GPUs |
|                  | CUDA             | 11.6.2                 | Apt            | Nvidia A100 GPUs require CUDA 11+ to work, so 10.x is not suitable |
|                  | CUDA toolkit     | 11.6.2                 | Apt            | Needed for `nvcc` command for cuDNN |
|                  | cuDNN            | 8.4.1.*-1+cuda11.6     | Apt            | Nvidia GPU deep learning library |
| Python           | Python           | 3.9.15                 | Apt            | Most widely used programming language for data science |
|                  | pip3             | 22.2.2                 | Apt            | Enable easy installation of 1000s of other data science, etc., packages. |
|                  | NumPy            | 1.23.4                 | pip3           | Handle arrays, matrices, etc., in Python |
|                  | SciPy            | 1.9.2                  | pip3           | Fundamental algorithms for scientific computing in Python |
|                  | Pandas           | 1.5.0                  | pip3           | De facto standard for data science data exploration/preparation in Python |
|                  | Cloudpickle      | 2.2.0                  | pip3           | Makes it possible to serialize Python constructs not supported by the default pickle module |
|                  | Matplotlib       | 3.6.1                  | pip3           | Widely used plotting library in Python for data science, e.g., scikit-learn plotting requires it |
|                  | Ipython          | 8.5.0                  | pip3           | Provides a rich architecture for interactive computing |
|                  | IPykernel        | 6.16.0                 | pip3           | Provides the IPython kernel for Jupyter. |
|                  | IPywidgets       | 8.0.2                  | pip3           | Interactive HTML widgets for Jupyter notebooks and the IPython kernel | 
|                  | Cython           | 0.29.32                | pip3           | Enables writing C extensions for Python |  
|                  | tqdm             | 4.64.1                 | pip3           | Fast, extensible progress meter |  
|                  | gdown            | 4.5.1                  | pip3           | Google drive direct download of big files |  
|                  | Pillow           | 9.2.0                  | pip3           | Python imaging library |  
|                  | seaborn          | 0.12.0                 | pip3           | Python visualization library based on matplotlib |
|                  | SQLAlchemy       | 1.4.41                 | pip3           | Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL |  
|                  | spaCy            | 3.4.1                  | pip3           | library for advanced Natural Language Processing in Python and Cython |  
|                  | nltk             | 3.7                    | pip3           | Natural Language Toolkit (NLTK) is a Python package for natural language processing |  
|                  | boto3            | 1.24.90                | pip3           | Amazon Web Services (AWS) Software Development Kit (SDK) for Python |  
|                  | tabulate         | 0.9.0                 | pip3           | Pretty-print tabular data in Python |  
|                  | future           | 0.18.2                 | pip3           | The missing compatibility layer between Python 2 and Python 3 |  
|                  | gradient         | 2.0.6                  | pip3           | CLI and Python SDK for Paperspace Core and Gradient |  
|                  | jsonify          | 0.5                    | pip3           | Provides the ability to take a .csv file as input and outputs a file with the same data in .json format |  
|                  | opencv-python    | 4.6.0.66               | pip3           | Includes several hundreds of computer vision algorithms |   
|                  | JupyterLab       | 3.4.6                  | pip3           | De facto standard for data science using Jupyter notebooks |
|                  | wandb            | 0.13.4                 | pip3           | CLI and library to interact with the Weights & Biases API (model tracking) |
| Machine Learning | Scikit-learn     | 1.1.2                  | pip3           | Widely used ML library for data science, generally for smaller data or models |
|                  | Scikit-image     | 0.19.3                 | pip3           | Collection of algorithms for image processing |
|                  | TensorFlow       | 2.9.2                  | pip3           | Most widely used deep learning library, alongside PyTorch |
|                  | torch            | 1.12.1                 | pip3           | Most widely used deep learning library, alongside TensorFlow |
|                  | torchvision      | 0.13.1                 | pip3           | Most widely used deep learning library, alongside TensorFlow |
|                  | torchaudio       | 0.12.1                 | pip3           | Most widely used deep learning library, alongside TensorFlow |
|                  | Jax              | 0.3.23                 | pip3           | Popular deep learning library brought to you by Google |
|                  | Transformers     | 4.21.3                 | pip3           | Popular deep learning library for NLP brought to you by HuggingFace |
|                  | Datasets         | 2.4.0                  | pip3           | A supporting library for NLP use cases and the Transformers library brought to you by HuggingFace |
|                  | XGBoost          | 1.6.2                  | pip3           | An optimized distributed gradient boosting library |
|                  | Sentence Transformers | 2.2.2             | pip3           | A ML framework for sentence, paragraph and image embeddings |

### Licenses

| Software              | License                | Source |
| ---------------       | -------------          | ------------- |
| CUDA 	                | NVidia EULA		  	 | https://docs.nvidia.com/cuda/eula/index.html |
| cuDNN                 | NVidia EULA            | https://docs.nvidia.com/deeplearning/cudnn/sla/index.html |
| JupyterLab            | New BSD      	         | https://github.com/jupyterlab/jupyterlab/blob/master/LICENSE |
| Matplotlib            | PSF-based      		 | https://matplotlib.org/stable/users/license.html |
| Numpy      	        | New BSD                | https://numpy.org/doc/stable/license.html |
| NVidia Docker         | Apache 2.0             | https://github.com/NVIDIA/nvidia-docker/blob/master/LICENSE |
| NVidia Driver         | NVidia EULA            | https://www.nvidia.com/en-us/drivers/nvidia-license/ |
| Pandas                | New BSD                | https://github.com/pandas-dev/pandas/blob/master/LICENSE |
| Pip3                  | MIT                    | https://github.com/pypa/pip/blob/main/LICENSE.txt |
| Python                | PSF                    | https://en.wikipedia.org/wiki/Python_(programming_language) |
| Scikit-learn          | New BSD                | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| Scikit-image          | New BSD                | https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt |
| TensorFlow            | Apache 2.0             | https://github.com/tensorflow/tensorflow/blob/master/LICENSE |
| PyTorch               | New BSD                | https://github.com/pytorch/pytorch/blob/master/LICENSE |
| Jax                   | Apache 2.0             | https://github.com/google/jax/blob/main/LICENSE |
| Transformers          | Apache 2.0             | https://github.com/huggingface/transformers/blob/main/LICENSE |
| Datasets              | Apache 2.0             | https://github.com/huggingface/datasets/blob/main/LICENSE |
| XGBoost               | Apache 2.0             | https://github.com/dmlc/xgboost/blob/master/LICENSE |
| Sentence Transformers | Apache 2.0             | https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE |
| SciPy                 | New BSD                | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| Cloudpickle           | New BSD                | https://github.com/cloudpipe/cloudpickle/blob/master/LICENSE |
| Ipython               | New BSD                | https://github.com/ipython/ipython/blob/main/LICENSE |
| IPykernel             | New BSD                | https://github.com/ipython/ipykernel/blob/main/COPYING.md |
| IPywidgets            | New BSD                | https://github.com/jupyter-widgets/ipywidgets/blob/master/LICENSE |
| Cython                | Apache 2.0             | https://github.com/cython/cython/blob/master/LICENSE.txt |
| tqdm                  | MIT                    | https://github.com/tqdm/tqdm/blob/master/LICENCE |
| gdown                 | MIT                    | https://github.com/wkentaro/gdown/blob/main/LICENSE |
| Pillow                | HPND                   | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| seaborn               | New BSD                | https://github.com/mwaskom/seaborn/blob/master/LICENSE.md |
| SQLAlchemy            | MIT                    | https://github.com/sqlalchemy/sqlalchemy/blob/main/LICENSE |
| spaCy                 | MIT                    | https://github.com/explosion/spaCy/blob/master/LICENSE |
| nltk                  | Apache 2.0             | https://github.com/nltk/nltk/blob/develop/LICENSE.txt |
| boto3                 | Apache 2.0             | https://github.com/boto/boto3/blob/develop/LICENSE |
| tabulate              | MIT                    | https://github.com/astanin/python-tabulate/blob/master/LICENSE |
| future                | MIT                    | https://github.com/PythonCharmers/python-future/blob/master/LICENSE.txt |
| gradient              | ISC                    | https://github.com/Paperspace/gradient-cli/blob/master/LICENSE.txt |
| jsonify               | MIT                    | https://pypi.org/project/jsonify/0.5/#data |
| opencv-python         | MIT                    | https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt |
| wandb                 | MIT                    | https://github.com/wandb/wandb/blob/master/LICENSE |


Information about license types:

Apache 2.0: https://opensource.org/licenses/Apache-2.0  
MIT: https://opensource.org/licenses/MIT  
New BSD: https://opensource.org/licenses/BSD-3-Clause  
PSF = Python Software Foundation: https://en.wikipedia.org/wiki/Python_Software_Foundation_License
HPND = Historical Permission Notice and Disclaimer: https://opensource.org/licenses/HPND
ISC: https://opensource.org/licenses/ISC

Open source software can be used for commercial purposes: https://opensource.org/docs/osd#fields-of-endeavor.


## Software not included

Other software considered but not included.

The potential data science stack is far larger than any one person will use so we don't attempt to cover everything here.

Some generic categories of software not included:

 - Non-data-science software
 - Commercial software
 - Software not licensed to be used on an available VM template
 - Software only used in particular specialized data science subfields (although we assume our users probably want a GPU)

| Category           | Software | Why Not |
| -------------      | ------------- | ------------- |
| Apache             | Kafka, Parquet | |
| Classifiers        | libsvm | H2O contains SVM and GBM, save on installs |
| Collections        | ELKI, GNU Octave, Weka, Mahout | |
| Connectors         | Academic Torrents | |
| Dashboarding       | panel, dash, voila, streamlit | |
| Databases          | MySQL, Hive, PostgreSQL, Prometheus, Neo4j, MongoDB, Cassandra, Redis | No particular infra to connect to databases |
| Deep Learning      | Caffe, Caffe2, Theano, PaddlePaddle, Chainer, MXNet | PyTorch and TensorFlow are dominant, rest niche |
| Deployment         | Dash, TFServing, R Shiny, Flask | Use Gradient Deployments |
| Distributed        | Horovod, OpenMPI | Use Gradient distributed |
| Feature store      | Feast | |
| Interpretability   | LIME/SHAP, Fairlearn, AI Fairness 360, InterpretML | |
| Languages          | R, SQL, Julia, C++, JavaScript, Python2, Scala | Python is dominant for data science |
| Monitoring         | Grafana | |
| NLP                | GenSim | |
| Notebooks          | Jupyter, Zeppelin | JupyterLab includes Jupyter notebook |
| Orchestrators      | Kubernetes | Use Gradient cluster|
| Partners           | fast.ai | Currently support a specific fast.ai runtime |
| Pipelines          | AirFlow, MLFlow, Intake, Kubeflow | |
| Python libraries   | statsmodels, pymc3, geopandas, Geopy, LIBSVM | Too many to attempt to cover |
| PyTorch extensions | Lightning | |
| R packages         | ggplot, tidyverse | Could add R if customer demand |
| Recommenders       | TFRS, scikit-surprise | |
| Scalable           | Dask, Numba, Spark 1 or 2, Koalas, Hadoop | |
| TensorFlow         | TF 1.15, Recommenders, TensorBoard, TensorRT | Could add TensorFlow 1.x if customer demand. Requires separate tensorflow-gpu for GPU support. |
| Viz                | Bokeh, Plotly, Holoviz (Datashader), Google FACETS, Excalidraw, GraphViz, ggplot2, d3.js | |