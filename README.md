## Code for the paper Staging Epileptogenesis with Deep Neural Networks
#### Published in Proceedings of the 11th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics

The architecture was inspired by the ResNet developed for [Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network](https://www.nature.com/articles/s41591-018-0268-3?source=techstories.org
) 

### Dependecies
The code was originally developed in tensorflow 1.8X. One need to adapt the code to tensorflow 2.0+

#### File organization
* exp_params.json: parameters for running the experiment, including file path, batch size, epochs, etc., 
* model_params.json: parameters for building models such as number of layers, kernel sizes, dropout rate, etc.
* main_EPG_classification.py: the main file to run the experiment.
* train.py: detailed training procedure including training, testing steps.
* dataio_EPG.py: helper functions regarding data loading, IO reading and writing, etc.
* plots_EPG.py: plot-related helper functions.

#### Data organization
<img width="407" alt="Screenshot 2022-11-07 at 18 32 40" src="https://user-images.githubusercontent.com/22638701/200376688-f6c0c338-d639-4da1-831c-469d48242fed.png">

#### How to use
> python main_EPG_classification.py
