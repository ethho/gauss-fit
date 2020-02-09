# Change Log
                                
### 9/19/2018

Added ability to "lock" parameters. Each slider now has a small 'Lock' button next to it that allows the user to fix a certain parameter at the constant defined by the slider value. Clicking the 'Lock' button will change its label to 'Float' in case the user wants to float that parameter again. Also added more documentation and cleaned up some spaghetti code. Updated to v0.1.1.

### 10/4/2018

Catches RuntimeErrors thrown by scipy.optimize. Handles by setting optim = guess and re-loading the fitter for the same column. Updated to v0.1.2. Started working on v0.1.3, which will calculate the difference function measurement-fit, then integrate along x. This should account for the area of the measurement function that is not accounted for by the fit function. Updated to v0.1.4.

### 10/5/2018

Fixed bug where optim params don't print as they should. Changed Interactive_Fitter.optim attr to list instead of array, so it's wayyy easier to manipulate now. The last value of the list printed into each column of the optim.txt file is the diff_area attr. Keep in mind that the scipy.optimize function will tend to minimize the diff_area if baseline is not locked. Updated to v0.1.5. 

### 10/8/2018

Made it so you don't have to call every input file "data.txt" This is dumb. Instead, gauss_fit will look for files ending in data.txt in the same folder as the gauss_fit.py script. It will also name the optim.txt file {prefix}optim.txt based on the {prefix}data.txt file name. Updated to v0.1.6.
