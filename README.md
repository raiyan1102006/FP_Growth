# FP-Growth
FP Growth is a popular algorithm for mining frequent itemsets from transaction databases. In this project, I have implemented the algorithm as specified in Chapter 6 of Han et al.’s book [1].

## What's in this Repository
This repository contains the adult dataset that I've used to test the FP Growth algorithm, the python code file, and a project report file. The Report.pdf file details my data preprocessing steps and the data structures I've used.

## Installation and Usage
I have used Python 3.6.3 to develop and test the code, and have used popular libraries such as numpy, pandas, math, itertools and json.

Once the libraries and environments are set up, the code can by run by executing 

```
python FP_growth.py
```

## Implementation

The program takes the dataset and min_sup (the minimum support threshold) as the input; and gives the frequent itemsets and their supports as the output. 

I have chosen a support of 23%. The algorithmic details can be found in [1], while the implementation details can be found in the [Report.pdf](https://github.com/raiyan1102006/Apriori/blob/master/Report.pdf) file.

## References
[1] Jiawei Han, Jian Pei, and Micheline Kamber. 2011. *Data Mining: Concepts and Techniques*. Elsevier.

