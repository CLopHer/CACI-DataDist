
# CACI Data Distillation

This project is a proof-of-concept test for using data distillation to reduce dataset size while maintain accuracy within DNN models. For this project specificly FashionMNIST was used as the dataset and the two primary models used were a simplied VGG and various ResNet NNs. 


## Dependencies
The dependencies contains a txt of all python libraries and installed versions. This file also include information regarding the CUDA drivers Here a image of the GPU used for all .py runs can be found. These have been included to aid in repoduction should the need arise. Python 3.9 was the primary language used for this project.
## .py vs .ipynb
Throughout the project we had experiment with both using python notebooks and python scripts. Most if not all .py scripts are just .ipynb files w/o the markdown cells. The primary purpose for this was for CUDA. Implementaions of .ipynb was done through Visual Studio Code(VSC) and .py was done in Visual Studion 2019(VS19). The reason for this was difficult(or lack of support) getting a notebook to run CUDA in VSC. When the files were made into .py and ran on VS19 there were zero issues and CUDA was able to be properly utilized
## Navigation
#### Pycache
Certain files made during implementation for usability, does not need to be interacted with

#### Data/FashionMNIST
Storage of the FMNIST. Two directories due some files having implementation issues in importing the dataset due to directory location

#### Experiments and Research
Here our VGG models and personal Experiments can be found. These include different implementations of a ResNet, KMeans subsetting, and Uniform subsetting.

#### Result
The outputs(synthesized images and terminal) of running main.py in Test Syn can be found here

#### Synthesis/Test Syn
implementation of synthesis algorithms for distillation

#### name.txt

Outputs of various files found within the repository
## Potential Future Work
Should this project continue to a next field session we belive we have provide with a good foundation for true distillation to take place. By the time we had created and understood our ResNet we had two weeks left and had dropped subsetting to attempt to focus on subsetting.

To see our internal report regarding this project please check:

https://cs-courses.mines.edu/csci370/FS2023S/Groups2023S.html

The final report should be posted under our advisors name in our CACI team section.
## Authors
Now that the first implementation is complete we, "The Client Doesn't Care What we Call Ourselves," would like to provide a proper introduction to ourselves

**Cesar Lopez Hernandez**

Major: Computer Science - Data Science

Hometown: Aurora, CO

Experience:

- Professional
  - Mines ITS MSC Lead Consultant
- Code
  - Python
    - TensorFlow
    - Numpy
    - Pandas
    - PyTorch
    - Matplotlib
    - Sklearn
    - KMeans
  - Java
  - C++
  - C

**Peter Gultom**

Computer Science / Data Science

Hometown: Aurora, Colorado

Work Experience: 
- Technical assistance for faculty

**Duan Nguyen**

Major: Computer Science

Hometown: Vietnam

Experience:

- Python: numpy
- C++
- Java

**Denisha Saviela**

Major: Computer Science - Data Science

Hometown: Indonesia

Work Experience:

- Professional
  - Tutoring Instructor
- Code
  - Python
  - Java
  - C++
  - C

## Contributing

Use this repository as you see fit, that being CACI or any potential group that may continue this project.

