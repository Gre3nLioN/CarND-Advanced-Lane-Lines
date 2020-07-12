import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = [0,0,0]
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = []
        #y values for detected line pixels
        self.ally = []
        # all radius_of_curvature
        self.all_radius_of_curvature = []

    def update(self, x, y, curvature, dist_middle):
        # less diference than 0.5 or first element
        if (np.mean(self.best_fit - np.polyfit(y, x, 2), axis=0) < 0.5 or len(self.allx) == 0):
            self.diffs = self.best_fit - np.polyfit(y, x, 2)
            self.detected = True
            self.recent_xfitted = x
            self.best_fit = np.average([self.best_fit, np.polyfit(y, x, 2)], axis=0)
            self.current_fit = np.polyfit(y, x, 2)
            self.radius_of_curvature = curvature
            np.concatenate((self.ally, y), axis=None)
            np.concatenate((self.allx, x), axis=None)
            self.bestx = np.average(self.allx, axis=0)
            self.all_radius_of_curvature.append(curvature)
            self.line_base_pos = dist_middle
        else:
            self.detected = False

    def get_average_curvature(self):
        if (len(self.all_radius_of_curvature) > 1):
            return np.average(self.all_radius_of_curvature[-60:], axis=0)
        else:
            return self.all_radius_of_curvature[-1]
