# Data-Science-Deep-learning-Sam_Jun
Learning deep learning this time!

## Caeser encryption function prediction

## TODO list
- [x] Caeser ciper and impliment deep learning model
- [ ] Try to play around with caeser ciper and deep learning input and output
* - [ ] input 3(d) -- output a(0)
* - [ ] input 3(d) -- output (1,0,0,...,0)
* - [ ] input (0,0,0,1,0,...,0) -- output a(0)
* - [ ] input (0,0,0,1,0,...,0) -- output (1,0,0,...,0)
* - [ ] input (0,0,0,1,0,...,0) -- output (1,0,0,...,0)
- [ ] Analyse this results
* For example, which model minimises the number of echos 
* Which model cannot reach the 100% accuracy? Why?
- [ ] Try to add more training data, see if things change
* If it is better, try to explain why
* If it is not, maybe 23 is enough?
* My prior: Maybe more data can make the training procedure faster? I don't know given the same input how doesthe neural network learn? Stop updating or **keep updating**?
- [ ] Try to hide some data, 
* For human, it is reasonable if we hide z, we are able to judge the one we cannot see (high probability)
* I tried it makes no sense so far. 98%?
- [ ] Try to given pairs of data, and change the structure of neural output and input
* How many we combination we have? **676** (no order). More deeperly, combining with more words (we know the number of combination is too large, we just want to test, if neural network could learn something unseen?
* Need a smart way to do this (consider to restruct the input and output as well as the order)<br>

Set|Input| Output|Seen
---|---|---|---|
Training| AB | DE |Yes
Training| CD | FG |Yes
Testing | BC | EF |No

Or 

Set|Input| Output|Seen
---|---|---|---|
Training| AB | DE |Yes
Testing | BA | ED |No

* Idea: 
- [ ] Watching videos and papers for logic programming
* To figure out logic programming in deep learning
Video<br>
[Deep learning course: Introduction to Deep Learning](https://www.youtube.com/watch?v=JN6H4rQvwgY)<br>
[Richard Evans: Inductive logic programming and deep learning I](https://www.youtube.com/watch?v=yD02DlZnHJw)<br>
[Learning Explanatory Rules from Noisy Data - Richard Evans, DeepMind](https://www.youtube.com/watch?v=_wuFBF_Cgm0&t=24s)<br>
Paper<br>
[LOGIC MINING USING NEURAL NETWORKS](https://arxiv.org/pdf/0804.4071.pdf)<br>
[First-order Logic Learning in Artificial Neural Networks](https://core.ac.uk/download/pdf/17294404.pdf)<br>
Python logic programming<br>
[PYKE](http://pyke.sourceforge.net/index.html)
## Issue for using keras
* Shape issue
https://github.com/MorvanZhou/tutorials/issues/30

## Python tips
* List to array
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
