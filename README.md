# gauss-fit
Matplotlib-based GUI for intuitive Gaussian curve fitting

Version: 0.1.6
Last updated: ENH 10/5/2018
Developed on Python 3.6.5

Fits Gaussian functions to a data set. No limit to the number of summed Gaussian components in the fit function. User can easily modify guess parameters using sliders in the matplotlib.pyplot window. Best fit parameters write to a tab-delimited .txt file called optim.txt. Data is entered into the program via a tab-delimited text file at ./data.txt, where the first column is the x values, and each successive column will be fit as y values. The data written to optim.txt is written as fit parameters for different y arrays in each column. Within a column, reading down the cells, values correspond to best fit parameters for baseline, area of Gaussian component 1 (a1), mean of component 1 (m1), standard deviation of component 1 (s1), area of component 2 (a2), and so forth. Closing matplotlib window will open another window to fit the next column over. Unfortunately, you cannot (yet) move backwards through columns.
