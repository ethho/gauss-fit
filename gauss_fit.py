#!/usr/bin/python

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import optimize

__version__ = "0.1.6"

# number of components initially and initial guess values
init_ncomp = 3
max_ncomp = 4
guess = [1700, 700, 1.2, .08, 200, 1.5, .18]

'''
                                gauss_fit.py

Version: 0.1.6
Last updated: ENH 10/5/2018
Developed on Python 3.6.5

Fits Gaussian functions to a data set. No limit to the number of summed
Gaussian components in the fit function. User can easily modify guess parameters
using sliders in the matplotlib.pyplot window. Best fit parameters write to
a tab-delimited .txt file called optim.txt. Data is entered into the program
via a tab-delimited text file at ./data.txt, where the first column is the x
values, and each successive column will be fit as y values. The data written to
optim.txt is written as fit parameters for different y arrays in each column.
Within a column, reading down the cells, values correspond to best fit
parameters for baseline, area of Gaussian component 1 (a1), mean of component 1
(m1), standard deviation of component 1 (s1), area of component 2 (a2), and so
forth. Closing matplotlib window will open another window to fit the next
column over. Unfortunately, you cannot (yet) move backwards through columns.

                                Change Log
                                
9/19/2018 - Added ability to "lock" parameters. Each slider now has a small
    'Lock' button next to it that allows the user to fix a certain parameter at
    the constant defined by the slider value. Clicking the 'Lock' button will
    change its label to 'Float' in case the user wants to float that parameter
    again. Also added more documentation and cleaned up some spaghetti code.
    Updated to v0.1.1.
10/4/2018 - Catches RuntimeErrors thrown by scipy.optimize. Handles by setting
    optim = guess and re-loading the fitter for the same column. Updated to
    v0.1.2. Started working on v0.1.3, which will calculate the difference
    function measurement-fit, then integrate along x. This should account for
    the area of the measurement function that is not accounted for by the fit
    function. Updated to v0.1.4.
10/5/2018 - Fixed bug where optim params don't print as they should. Changed
    Interactive_Fitter.optim attr to list instead of array, so it's wayyy easier
    to manipulate now. The last value of the list printed into each column of
    the optim.txt file is the diff_area attr. Keep in mind that the
    scipy.optimize function will tend to minimize the diff_area if baseline
    is not locked. Updated to v0.1.5. 
10/8/2018 - Made it so you don't have to call every input file "data.txt" This
    is dumb. Instead, gauss_fit will look for files ending in data.txt in the
    same folder as the gauss_fit.py script. It will also name the optim.txt file
    {prefix}optim.txt based on the {prefix}data.txt file name. Updated to v0.1.6.
'''

def gaussian(x, baseline, area, mean, std):
    return (area/(std*np.sqrt(2*np.pi)))*np.exp(-(x - mean)**2/(2*std**2)) + baseline

def three_gaussians(x, baseline, a1, m1, s1, a2, m2, s2, a3, m3, s3):
    return (gaussian(x, 0, a1, m1, s1) +
            gaussian(x, 0, a2, m2, s2) +
            gaussian(x, 0, a3, m3, s3) + baseline)

def four_gaussians(x, baseline, a1, m1, s1, a2, m2, s2, a3, m3, s3, a4, m4, s4):
    return (gaussian(x, 0, a1, m1, s1) +
            gaussian(x, 0, a2, m2, s2) +
            gaussian(x, 0, a3, m3, s3) +
            gaussian(x, 0, a4, m4, s4) + baseline)

def five_gaussians(x, baseline, a1, m1, s1, a2, m2, s2,
                   a3, m3, s3, a4, m4, s4, a5, m5, s5):
    return (gaussian(x, 0, a1, m1, s1) +
            gaussian(x, 0, a2, m2, s2) +
            gaussian(x, 0, a3, m3, s3) +
            gaussian(x, 0, a4, m4, s4) + 
            gaussian(x, 0, a5, m5, s5) + baseline)

def two_gaussians(x, baseline, a1, m1, s1, a2, m2, s2):
    return three_gaussians(x, baseline, a1, m1, s1, a2, m2, s2, 0, 0, 1)

def pad_zeros(arr, shape):
    '''
    Given array-like `arr`, returns a new array with `shape`. Axes in `shape`
    that are longer than those in `arr` are filled with 0 int type.
    '''
    new = np.zeros_like(arr)
    new[:arr.shape[0]] = arr
    return new
# end function pad_zeros

def fit_n_gaussians(x, y, n, guess, bounds=[]):
    '''
    Performs fit for n gaussian components given arrays x and y. Returns tuple
    like (optim, err, gauss_func).
    
    Arguments:
        array x - x values to fit
        array y - y values to fit
        int n - number of guassian components to fit to x,y data
        list guess - list of guess values
        list bounds[=[]] - list of bounds on optimized parameters. Formatted as
            [[min,max],[min,max],[min,max]]. Min can equal max. If null,
            defaults to no bounds for all parameters. 
    '''
    
    # set default bounds value
    if not bounds:
        bounds = (np.array([-np.inf] * len(guess)),
                       np.array([np.inf] * len(guess)))
    elif len(bounds) == len(guess):
        # reformat bounds into tuple of arrays. Scaling by ~1 is necessary
        # to satisfy min strictly less than max requirement
        bounds = (np.array([x[0]*0.999 for x in bounds]),
                  np.array([x[1]*1.001 for x in bounds]))
    else: # bad shape
        raise Exception(
            "`bounds` must be a 2-element tuple of arrays each " +
            "sharing the same shape as guess. Guess list has " +
            "length {} and bounds list has length {}".format(len(guess),
                                                             len(bounds)))
    
    # get the Gaussian function with n components
    if n == 1:
        gauss_func = gaussian
    elif n == 2:
        gauss_func = two_gaussians
    elif n == 3:
        gauss_func = three_gaussians
    elif n == 4:
        gauss_func = four_gaussians
    elif n == 5:
        gauss_func = five_gaussians
    else:
        raise Exception("n must be between 1 and 5")

    # fit function to x and y the old way using scipy.optimize.leastsq
    
    #errfunm = lambda p, x, y: (gauss_func(x, *p) - y)**2
    #optim, success = optimize.leastsq(errfunm, guess[:], args=(x, y))
    #print(optim)
    #err = np.sqrt(errfunm(optim, x, y)).sum()
    
    # new way: fit x and y using scipy.optimize.curve_fit, which allows
    # for parameter bounds
    try:
        optim, cov = optimize.curve_fit(
            gauss_func, x, y, p0 = guess[:], method = 'trf', bounds = bounds)
    except RuntimeError as e: # thrown by scipy.optimize
        #print("scipy.optimize threw RuntimeError. Setting optimized params to" +
        #      " guess values.")
        #print(e)
        return (np.array(guess), -1, gauss_func, np.zeros_like(x))
    # calculate y values for the difference function measurement-fit
    diff_y = y - gauss_func(x, *optim)
    
    # calculate total residual error
    errfunm = lambda p, x, y: (gauss_func(x, *p) - y)**2
    err = np.sqrt(errfunm(optim, x, y)).sum()
    
    return (optim, err, gauss_func, diff_y)
    
# end function fit_n_gaussians

class InteractiveFitter(object):
    
    def __init__(self, x, y, guess=[], init_ncomp=1, max_ncomp=5, title=""):
        '''
        Instantiates an InteractiveFitter instance. Using matplotlib.pyplot,
        shows the current measurement curve, Gaussian calculated from initial
        guess values, and the best fit. Guess values can be changed with
        sliders, interactively updating the best-fit function when the
        'Try Fit' button is pressed. Residual error is also shown. User can
        change the number of Gaussian components up to self.max_ncomp. Call
        plotfit to pop up the fitter window.
        
        Arguments:
            array x - x values to fit
            array y - y values to fit
            list guess - guess parameters. First element is always the baseline.
                Element (n*3) is the area under Gaussian component n. Likewise,
                elements (1+(n*3)) and (2+(n*3)) are the mean and standard
                deviation guess values for the nth Gaussian component. The
                Fitter will automatically extend or truncate the guess
                paremeters based on the number of components it is currently
                fitting.
            init_ncomp[=1] - initial number of components to fit
            max_ncomp[=5] - soft limit to the max integer value of the
                '# components' slider
            str title[=""] - title of the plot
        '''
        assert 0 < init_ncomp <= max_ncomp
        
        self.x = x
        self.y = y
        self.ncomp = init_ncomp
        self.max_ncomp = max_ncomp
        self.title = title
        
        # generate automatic guess parameters based on what we normally
        # see in gel images
        if guess:
            self.guess = guess
        else:
            self.guess = [np.min(self.y), 1000, 1.0, 0.1]
        
        # initial guess list
        self.init_guess = self.guess
        
    # end method __init__
    
    def update_fit(self, val):
        '''
        Update fit and guess curves on button event. Closes plot and
        re-instantiates, re-calculating best fit using the updated guesses.
        '''
        
        # update ncomp
        self.ncomp = int(self.sncomp.val)
        # re-render plot
        plt.close('all')
        self.plotfit() 
        
    def update_guess(self, val):
        '''
        Only update guess curve. Called upon slider change. 
        '''
        
        # update guess values
        self.guess = [self.sbsl.val]
        for i in range(1, self.ncomp+1):
            self.guess.append(getattr(self, "sarea{}".format(i)).val)
            self.guess.append(getattr(self, "smean{}".format(i)).val)
            self.guess.append(getattr(self, "sstd{}".format(i)).val) 
        self.g.set_ydata(self.gauss_func(self.x, *self.guess))
        self.fig.canvas.draw_idle()
    
    def update_button(self, event):
        '''
        Only update bounds. Called upon a 'Lock' button press.
        '''
        
        # find the button that was pressed according to its ax attr
        found = False
        for bname in [x for x in dir(self) if x[:5] == "block"]:
            if getattr(self, bname).ax == event.inaxes:
                found = True
                break
        # error if button was not found. Definitely a developer error and
        # nothing that should pop up for the user.
        if not found:
            raise Exception("Could not propogate backtrace to Button object.")
        # change Button attrs based on whether it is pressed or not
        if getattr(self, bname).pressed: # button is depressed
            facecolor = 'lightgoldenrodyellow'
            textcolor = 'black'
            label = 'Lock'
            pressed = False
        else: # button is not depressed
            facecolor = 'black'
            textcolor = 'white'
            label = 'Float'
            pressed = True
        
        # set facecolor, textcolor, label and pressed attr as appropriate
        getattr(self, bname).ax.set_facecolor(facecolor)
        getattr(self, bname).color = facecolor
        getattr(self, bname).label.set_text(label)
        getattr(self, bname).label.set_color(textcolor)
        setattr(getattr(self, bname), "pressed", pressed)
        self.fig.canvas.draw_idle()
        
    # end method update_button
    
    def plotfit(self):
        '''
        Handler for the interactive fitter. Adjusts for changes in self.ncomp,
        re-calculates best fit parameters (self.optim), and instantiates the
        pyplot figure. Also calls _slowsliders. Upon pyplot window close,
        returns fit parameters and residual error as a tuple.
        
        Arguments:
            none
        Returns:
            tuple - (array(parameters), int(residual error))
        '''
        
        # rotating color list for matplotlib
        color_lst = ['brown','teal','pink','yellow','purple','orange']
        
        # get expected length of guess list, and trim or extend the guess
        # list if necessary (in case user changed number of components)
        len_guess = 1 + self.ncomp*3
        while len_guess > len(self.guess):
            self.guess.extend(self.guess[1:4])
        if len_guess < len(self.guess):
            self.guess = self.guess[:len_guess]
            
        # Generate bounds list. Looks at the pressed attrs for each lock Button
        # and generates a list of bounds like self.guess, except each element
        # is a len=2 list containing the allowed min and max values for that
        # parameter.
        if getattr(getattr(self, "blocksbsl", False), "pressed", False):
            self.bounds = [[self.sbsl.val] * 2]
        else:
            self.bounds = [[-np.inf,np.inf]]
        # loop through each parameter for each gaussian component. Determine
        # if the lock button for that parameter is pressed (or if it exists
        # at all), and set self.bounds for that element to the slider value.
        # Otherwise, bounds are infinite and practically nonexistent.
        for i in range(1, self.ncomp+1):
            for pname in ("area", "mean", "std"):
                if getattr(getattr(self, "block{}{}".format(pname,i), False),
                           "pressed", False):
                    self.bounds.append([getattr(self, "s{}{}".format(pname,i)).val] * 2)
                else:
                    self.bounds.append([-np.inf,np.inf])
        
        # fit data to n Gaussian components
        optim_arr, self.err, self.gauss_func, self.diff_y = fit_n_gaussians(
            self.x, self.y, self.ncomp, self.guess, self.bounds)
        # convert optim to list
        self.optim = list(optim_arr)
        # error message if RuntimeError threw in fit_n_gaussians
        if self.err == -1:
            print("scipy.optimize threw RuntimeError while processing " +
                  "{}. Setting optimized ".format(self.title) +
                  "params to guess values.")
            self.optim = self.guess
        # integrate the difference function over x
        self.diff_area = np.trapz(self.diff_y, self.x)
        
        #               PLOT EVERYTHING    
        
        # matplotlib instantiation and plot measurement curve
        self.fig, ax = plt.subplots(figsize=(15,10))
        ax.plot(self.x, self.y, lw=3.5, c='g', label='Measurement')
        
        # plot the fit curve
        self.f, = ax.plot(self.x, self.gauss_func(self.x, *self.optim), lw=2.5,
                          c='r', label='Fit')
        # plot the guess function
        self.g, = ax.plot(self.x, self.gauss_func(self.x, *self.guess), lw=1.5,
                          c='b',label='Guess'.format(self.ncomp), ls="--")
        # plot the difference function
        ax.plot(self.x, self.diff_y, lw=1, c='purple', label='Difference')
        # plot each component of the fit function
        for i in range(self.ncomp):
            ax.plot(self.x, gaussian(self.x, self.optim[0],
                                     *self.optim[((i*3)+1):((i*3)+4)]), lw=1, 
                    label='Fit (comp. #{})'.format(i+1), ls="--",
                    c=color_lst[i%6])
        # adjust subplot size to accommodate sliders
        self.fig.subplots_adjust(left=0.25, bottom=0.12+self.ncomp*0.08)
        # instantiate sliders
        self._showsliders()
        # generate re-fit button
        axrefit = plt.axes([0.025, 0.1, 0.1, 0.04])
        brefit = Button(axrefit, 'Try Fit', color='lightgoldenrodyellow')
        brefit.on_clicked(self.update_fit)
        # generate text box showing residual error and fit parameters
        eb_str = "Residual error: {:.1f}\n".format(self.err) + \
        "Baseline: {:.1f}".format(self.optim[0])
        for i in range(1, len(self.optim)):
            eb_str += "\n{1}: {0:.2f}".format(self.optim[i],
                [x+str(1+((i-1)//3)) for x in ['Area','Mean','STD']][(i-1)%3])
        # add the self.diff_area stat to the end
        eb_str += "\nDiff. area: {0:.2f}".format(self.diff_area)
        self.eb = self.fig.text(0.025, 0.4, eb_str, bbox=dict(
            facecolor='lightgoldenrodyellow', alpha=0.2))
        # generate text box showing column name
        self.fig.text(0.5, 0.95, self.title, bbox=dict(
            facecolor='lightgoldenrodyellow', alpha=0.2))
        # show legend and plot
        ax.legend(loc='best')
        #print(self.optim)
        plt.show()
        return (self.optim, self.err)
        
    # end method trackfit
    
    def _showsliders(self):
        '''
        Instantiate slider objects in pyplot.
        '''
        
        scolor = 'lightgoldenrodyellow' # default slider color
        ss = 0.1 # scaling factor for slider heights
        ival = self.guess # initial slider values are same as guess values
        
        # slider for baseline
        axbsl = plt.axes([0.22, 0.05, 0.65, 0.015], facecolor=scolor)
        # make sure starting value is between the min and max y values
        bsl_min = np.min(self.y)
        if bsl_min < 0:
            bsl_min = 0
        if not (bsl_min <= ival[0] <= np.max(self.y)):
            ival[0] = np.min(self.y)
        # Slider object
        self.sbsl = Slider(axbsl, "Baseline", bsl_min, np.max(self.y/2.),
                           valinit=ival[0])
        # event forwarding
        self.sbsl.on_changed(self.update_guess)
        
        # Lock button for baseline
        
        # Attempt to retrieve pressed value of a lock Button if it
        # existed before. Otherwise, set pressed to False.
        block_pressed = getattr(getattr(self, "blocksbsl", False),
                                "pressed", False)
        # color settings if Button was already pressed
        if block_pressed:
            block_facecolor, block_label, block_textcolor = 'black', 'Float', \
                                                            'white'
        else: # new Button or Button was not pressed
            block_facecolor, block_label, block_textcolor = scolor, 'Lock', \
                                                            'black'
        # Instantiate axes
        self.axlocksbsl = plt.axes([0.94, 0.05, 0.03, 0.015])
        # instantiate Button object
        self.blocksbsl = Button(
            self.axlocksbsl,
            block_label, color=block_facecolor)
        # lock button font size and text color
        self.blocksbsl.label.set_fontsize(6)
        self.blocksbsl.label.set_color(block_textcolor)
        # set pressed value for Button equal to True if pressed before
        self.blocksbsl.pressed = block_pressed
        # lock button click event
        self.blocksbsl.on_clicked(self.update_button)
        
        
        # slider for ncomp
        axncomp = plt.axes([0.22, 0.07, 0.65, 0.015], facecolor=scolor)
        self.sncomp = Slider(axncomp, "# components", 1, self.max_ncomp,
                             valstep=1, valinit=self.ncomp)
        
        # generate 3 param sliders for each gaussian component
        for i in range(1, self.ncomp+1):
            # define slider parameters
            # [min, max, and init] values for sliders based on ncomp, and
            # ypos: distance from bottom of window
            sparams = {
                'area': {
                    'range': [0., abs(ival[1+(i-1)*3]*2), ival[1+(i-1)*3]],
                    'ypos': 1.2
                    },
                'mean': {
                    'range': [np.min(self.x), np.max(self.x), ival[2+(i-1)*3]],
                    'ypos': 1.4
                    },
                'std': {
                    'range': [0., abs(ival[3+(i-1)*3]*2), ival[3+(i-1)*3]],
                    'ypos': 1.6
                }}
            # iterate through "area", "mean", and "std"
            for pname in ("area", "mean", "std"):
                # define slider axis
                setattr(self, "ax{}{}".format(pname, i),plt.axes(
                    [0.22,(sparams[pname]['ypos']*ss)+(i-1)*(ss*.7),
                     0.65, ss/10.], facecolor=scolor))
                # instantiate Slider object
                setattr(self, "s{}{}".format(pname, i), Slider(
                    getattr(self, "ax{}{}".format(pname, i)),
                    "{}{}".format(pname,i), sparams[pname]['range'][0],
                    sparams[pname]['range'][1],
                    valinit=sparams[pname]['range'][2]))
                # trigger update_guess event on slider change
                getattr(self, "s{}{}".format(pname, i)).on_changed(
                    self.update_guess)
                
                # 'Lock' Button next to slider. Prevents a param from floating
                # during optimization
                
                # Attempt to retrieve pressed value of a lock Button if it
                # existed before. Otherwise, set pressed to False.
                block_pressed = getattr(getattr(self, "block{}{}".format(pname, i),
                                             False), "pressed", False)
                # color settings if Button was already pressed
                if block_pressed:
                    block_facecolor, block_label, block_textcolor = 'black', 'Float', 'white'
                else: # new Button or Button was not pressed
                    block_facecolor, block_label, block_textcolor = scolor, 'Lock', 'black'
                # Instantiate axes
                setattr(self, "axlock{}{}".format(pname, i), plt.axes(
                    [0.94, (sparams[pname]['ypos']*ss)+(i-1)*(ss*.7), 0.03,
                     0.01]))
                # instantiate Button object
                setattr(self, "block{}{}".format(pname, i), Button(
                    getattr(self, "axlock{}{}".format(pname, i)),
                    block_label, color=block_facecolor))
                # lock button font size and text color
                getattr(self, "block{}{}".format(pname, i)).label.set_fontsize(6)
                getattr(self, "block{}{}".format(pname, i)).label.set_color(
                    block_textcolor)
                # set pressed value for Button equal to True if pressed before
                getattr(self,
                        "block{}{}".format(pname, i)).pressed = block_pressed
                # lock button click event
                getattr(self, "block{}{}".format(pname, i)).on_clicked(
                    self.update_button)
                
    # end method _showsliders
    
def main_handler():
    '''
    Main handling function for gauss_fit.py
    '''
    
    # get file names ending in data.txt. Skip if there's a
    # {prefix}optim.txt file
    in_fp = ""
    for poss_in in os.listdir("./"):
        try:
            prefix, suffix = poss_in[:-8], poss_in[-8:]
            optim_fp = "{}optim.txt".format(prefix)
            if suffix == "data.txt" and prefix[0] != "." and (not os.path.isfile(optim_fp)):
                in_fp = prefix + suffix
                break
        except IndexError:
            continue
    # all data.txt files have been analyzed
    if not in_fp:
        print("No *data.txt files found without existing *optim.txt files.")
        return
    
    # input data file
    data = np.genfromtxt(in_fp)
    
    # loop through each column after the first one (x values)
    optims, errs = [], []
    max_len = 0
    for colid in range(1, data.shape[1]):
        y = data[:,colid]
        # instantiate and run matplotlib fitter
        fitter = InteractiveFitter(data[:,0], y, guess, init_ncomp, max_ncomp,
                                   "{} column {}".format(in_fp, colid+1))
        optim, res_err = fitter.plotfit()
        optim.append(fitter.diff_area)
        optims.append(optim)
        errs.append(res_err)
        # increase max list length if necessary
        if len(optim) > max_len:
            max_len = len(optim)
        #print(optim)
    
    # resize all optim lists to same shape and import into ndarray for csv write
    optims_arr = np.array([(l + [0.] * max_len)[:max_len] for l in optims])
    try:
        optims_arr = np.swapaxes(optims_arr, 0, 1)
    except ValueError as e:
        #print(optims_arr)
        raise e
    
    # save as tab-delimited text file
    np.savetxt('{}optim.txt'.format(prefix), optims_arr, delimiter='\t')
    
# end function main_handler

if __name__ == '__main__':
    main_handler()