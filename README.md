#Finding Optimal learing rate with Bayesian Optimization 

Optimizing the learning rate of a RESNET-9 model for KMNIST Data. 

Folder Structure:
```
.
├── base
│   ├── __init__.py
│   ├── optimizers.py
│   ├── __pycache__
│   ├── regressors.py
│   └── utils.py
├── datasets
│   └── KMNIST
├── docs
│   ├── dataloaders.html
│   ├── models.html
│   ├── optimizers.html
│   ├── regressors.html
│   ├── trainers.html
│   └── utils.html
├── logs
│   └── log_19_53_10_01_2021.log
├── main.py
├── NeuralNet
│   ├── dataloaders.py
│   ├── __init__.py
│   ├── models.py
│   ├── __pycache__
│   └── trainers.py
├── README.md
├── requirements.txt
└── results
    ├── bo_result_0.pdf
    ├── bo_test.pdf
    └── gpr_test.pdf


```

Usage:

    Refer main.py for example usage. logs will be collected in logs folder.  
    
Docs:
    docs directory has documentation for all modules. 

Results: 
    
    bo_test.pdf and gps_test.pdf have sample gaussian regreeion output and Bayesian optimization on     
    a sample function. 

    The results folders have pdf files bo_result_0-5.pdf files for different xi([0.001, 0.01, 0.05, 0.1, 1, 2])
    Bayesian optimizations of learning rate on KMNISTResNet Model. 
     bo_result_6.pdf is with xi=0.1 and range learning rate range (0,1). 
    The pdf contains the graph of initial learning range test, activation function and 
    gaussian regression graphs for each function evaluation in Bayesian optimization.
    Finally, a graph to observe the convergence of Bayeisan optimization and the
    distance between current lambda to next lambda to observe exploration vs exploitation. 

Final Result:

    File: bo_result_3.pdf
    Good xi value: 0.1
    Optimal learning Rate: 
        learning_rate: 0.07596445
        training loss: 0.17878852749907723                     , 
        validation loss: 0.19897371530532837
