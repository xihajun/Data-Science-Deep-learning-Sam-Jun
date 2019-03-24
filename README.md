## Report
Our final code can be find in [./project summary/project.ipynb](https://github.com/xihajun/Data-Science-Deep-learning-Sam-Jun/tree/master/project%20summary)

# Install requirements

All the requirements are in requirements.txt file, you can either use `pip install -r requirements.txt` to get the package installed or do some set up things as follows:

## Set up

If you have Jupyter in your requirements.txt and you activate that environment (I recommend Virtualenv), install, and run Jupyter, you'll have all the specific versions you want. So:

     python3 -m venv venv
     source venv/bin/activate #(different command on Windows)
     pip install -r requirements.txt
     jupyter notebook
## Requirement
     pip install -r requirements.txt
     
     
# Data-Science-Deep-learning-Sam_Jun

As Deep learning is able to learn all the function, this time we are trying to make our model learning the encryption function.
Firstly, let start at the basic caeser encryption!(as the algorithm is determined, it might be too easy for neural networks to learn it) 

Our plan is to make model as simple as possible in order to find out the deep meanning in deep learning model. We will analyse the accuracy and loss curve as we change different hyperparameters (eg, num of layers, input output architechture, optimisor, loss function and so on). When we reduce our num of layer and neurons into a small number, we are able to use the definition of neural network to **calculate the prediction** result by using our _weights and bias_ given input. See the weights distribution if there exist an weight that dominate the model?

Also, we are trying to figure out if the DL algorithms are robust to random errors in the training set, so we add noise for our labels and get an **Accuracy Curve**(it should be the general accuracy curve and not general)

In addtion, we are going to hide some data, as we can see this function is an one-to-one correspondence, but when we hide one letter(eg, z) if the deep learning algrithm can learned it should be follow the same rule?**(We will try this in 1\*20\*20*1 neural network.)**

Rather than only consider one letter once, we are also care about sequence. If we given sequence, can our model learn the rule as well? We will start at sequence with length 2. More deeply, if we given label AB can it learned BA?

After that, we are trying to use **hand-writing letter** image to train a neural to learn it and generate it. Of course, see if it can decoded it once given correct labels.

# Caeser
## TODO list
- [x] **Caeser ciper and impliment deep learning model**
* Caeser encryption function prediction
- [x] **Try to play around with caeser ciper and deep learning input and output**
    - [x] input 3(d) -- output a(0)
        When dimensions of input and output are 1, the model cannot learn things well even we tried a lot of different loss function and activation function. For the data without XYZ which means when the system doesn't take the shift 23-0, 24-1, 25-2 into account the result will be better, but bascially it just learned the linear function. The possible solution is rather just use the integer data, we can add more data between integers and guide the model learn things better.
    - [x] input 3(d) -- output (1,0,0,...,0)
        Similarly, the one input cannot get a better preformance.
    - [x] input (0,0,0,1,0,...,0) -- output a(0)
    - [x] input (0,0,0,1,0,...,0) -- output (1,0,0,...,0)
        The last one, contains a lot of data information, which transfer the question into a classification problem and also can get a quite good result.
- [ ] **Analyse this results**

   For example, which model minimises the number of echos 
   
   Which model cannot reach the 100% accuracy? Why?
   
- [ ] **Using unseen data 26-100**

    The model has seen 0-26)<br>
    Try to find the weight in this model<br>
    Try to visualize it by using tensorboard<br>
- [x] **Try to add more training data, see if things change**<br>
    **It reduces the num of epochs.**
   
    ~~If it is better, try to explain why<br>~~
    ~~If it is not, maybe 23 is enough?<br>~~
* My prior: Maybe more data can make the training procedure faster? I don't know given the same input how doesthe neural network learn? Stop updating or **keep updating**?

- [ ] **Try to hide some data**
* For human, it is reasonable if we hide z, we are able to judge the one we cannot see (high probability)
* I tried it makes no sense so far. 98%?
- [x] **Try to given pairs of data, and change the structure of neural output and input**
* How many we combination we have? **676** (no order). More deeperly, combining with more words (we know the number of combination is too large, we just want to test, if neural network could learn something unseen?
* Need a smart way to do this (consider to restruct the input and output as well as the order)<br>

<table>
<tr><th>Example 1 </th><th>Example 2</th></tr>
<tr><td>

Set|Input| Output|Seen
---|---|---|---|
Training| AB | DE |Yes
Training| CD | FG |Yes
Testing | BC | EF |No

</td><td>


Set|Input| Output|Seen
---|---|---|---|
Training| AB | DE |Yes
Testing | BA | ED |No

</td></tr> </table>

* Idea: (1,1,0,0,0) or (1,2,0,0,0)
* Defined an position matrix which considers the order
**In addtion, we tried 3 letters combinations.**


- [x] **Add noise for the dataset**
* Given 0.1 0.2 errors, see the prediction results
* plot the prediction acc

The DL algorithm has a rubost property for random mislabelled data especially when the data size is big. However, for the data mislabelled on system arranged (for example, set all the cats as dogs, DL model doesn't have this property anymore.)

- [ ] **Watching videos and papers for logic programming**

    To figure out logic programming in deep learning<br>
    Video<br>
    [Deep learning course: Introduction to Deep Learning](https://www.youtube.com/watch?v=JN6H4rQvwgY)<br>
    [Richard Evans: Inductive logic programming and deep learning I](https://www.youtube.com/watch?v=yD02DlZnHJw)<br>
    [Learning Explanatory Rules from Noisy Data - Richard Evans, DeepMind](https://www.youtube.com/watch?v=_wuFBF_Cgm0&t=24s)<br>
    Paper<br>
    [LOGIC MINING USING NEURAL NETWORKS](https://arxiv.org/pdf/0804.4071.pdf)<br>
    [First-order Logic Learning in Artificial Neural Networks](https://core.ac.uk/download/pdf/17294404.pdf)<br>
    Python logic programming<br>
    [PYKE](http://pyke.sourceforge.net/index.html)

- [ ] **Try to use ILD to guide our neural system**

    How? Set rules, recurrent?

- [ ] **Try to find new encryption function to learn more complex stuff**

    key=[1,2,3,4]? 
    
    Consider the input and output
    
    
   
## Issue - keras
~~* [fixed] Shape issue
https://github.com/MorvanZhou/tutorials/issues/30~~

## Python tips
* List to array(Clever)
```{python}
from keras.utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)
```
* List to array(Stupid)
```{python}
variable_name = np.array(yourlist_name)
```
* `with` in python
http://linbo.github.io/2013/01/08/python-with

## Code reference
* Caeser ciper
https://gist.github.com/jameslyons/8701593



## Reference
[25 Must Know Terms & concepts for Beginners in Deep Learning](https://www.analyticsvidhya.com/blog/2017/05/25-must-know-terms-concepts-for-beginners-in-deep-learning/#)
