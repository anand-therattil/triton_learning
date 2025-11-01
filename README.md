# triton_learning
Understanding Triton using simple functions.
Added following files 
- Mathematical Files (comparision between triton language addition vs torch vs cpu python)
    - add.py
    - softmax.py 
- model files (Testing of a simple model to triton)
    - test.py 
    - client.py 


Result of the mathematical files 
- Softmax
```
| Length      | Triton (ms) | Python (ms) | TorchScript (ms) |
|-------------|-------------|-------------|------------------|
| 100,000     | 97.648859   | 1.092935    | 3.694129         |
| 1,000,000   | 0.096846    | 9.861231    | 0.019026         |
| 10,000,000  | 0.084472    | 96.633530   | 0.015140         |
| 50,000,000  | 0.077987    | 484.304786  | 0.019765         |
| 100,000,000 | 0.095201    | 999.774003  | 0.018692         |
| 200,000,000 | 0.093699    | 2326.059008 | 0.017548         |
| 500,000,000 | 0.120807    | 7268.388677 | 0.022817         |
```
- ADDTION
```
| Length      | Triton (ms) | Python (ms) | TorchScript (ms) |
|-------------|-------------|-------------|------------------|
| 100,000     | 29.621100   | 1.400948    | 0.010347         |
| 1,000,000   | 0.047755    | 0.007677    | 0.006914         |
| 10,000,000  | 0.067735    | 0.007558    | 0.006723         |
| 50,000,000  | 0.075006    | 0.007558    | 0.006819         |
| 100,000,000 | 0.080967    | 0.007319    | 0.006485         |
| 200,000,000 | 0.082397    | 0.010967    | 0.006890         |
| 500,000,000 | 0.106192    | 0.010872    | 0.006342         |
 
```

BLOCK SIZE: input_size = (10000000,)
```
| Block Size | Time (ms)  |
|------------|------------|
| 256        | 31.697369  |
| 512        | 0.169826   |
| 1024       | 0.162768   |
| 2048       | 0.168514   |
| 4096       | 0.156617   |
| 8192       | 0.173020   |
```

OBSERVATION 
- for simple mathematics like +,-,*,/ the cpu speed and torch script speed is great as they are already optimized 
- for complex task like softmax triton is good but torchscript has an optimzed version so the perfomance is better 
- if there is complex math problem present the triton will surely be perfoming better 
- the BLOCK_SIZE effect are getting similar after increasing it to 1024 after testing for 3 iteration for softmax (Result show is for only 1 iteration)