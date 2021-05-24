import os
import sys
import arcade
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# Graphics constants
SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 800
SCREEN_TITLE  = "Shape Fitter"
FPS = 60

class MyGame(arcade.Window):

    def __init__(self, shape, alpha):
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.SKY_BLUE)

        # List of points so far
        self.points = None

        # List of best parameters so far
        if shape == 'c' or shape == 'circle':
            self.params = np.zeros(3)
        elif shape == 'l' or shape == 'line':
            self.params = np.zeros(2)

        # Store shape of interest
        self.shape = shape

        max_iter = 1e7

        # Create regression objects for later
        self.regressors = [
            LinearRegression(fit_intercept=False), 
            Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter), 
            Ridge(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        ]

        # Colors for reach type of regression
        self.colors = [arcade.color.RED, arcade.color.GREEN, arcade.color.BLACK]

    
    def draw_legend(self):

        x_legend = SCREEN_WIDTH - 150
        y_legend = SCREEN_HEIGHT - 100
        for col, reg in zip(self.colors, self.regressors):
            name = str(type(reg)).split(".")[-1][:-2]

            arcade.draw_text(name, x_legend, y_legend, col, 10)
            y_legend -= 0.02 * SCREEN_HEIGHT


    def on_draw(self):
        # This command has to happen before we start drawing
        arcade.start_render()

        # Draw the legend
        self.draw_legend()

        # Go through each type of regression
        if self.points is not None and len(self.points) > 3:
            for regressor, col in zip(self.regressors, self.colors):
                params = regressor.coef_
                if self.shape == 'c' or self.shape == 'circle':
                    # Find circle center and radius
                    xc = -params[0] / 2
                    yc = -params[1] / 2
                    r = np.sqrt(xc ** 2 + yc ** 2 + params[2])

                    # Draw best circle so far
                    arcade.draw_circle_outline(xc, yc, r, col, 1)
                elif self.shape == 'l' or self.shape == 'line':
                    # Draws line given y = mx + b (m and b)
                    m, b = params
                    x0 = -1
                    x1 = SCREEN_WIDTH + 1
                    y0 = m * x0 + b
                    y1 = m * x1 + b

                    # Draw best line so far
                    arcade.draw_line(x0, y0, x1, y1, col, 1)

        # Draw all points
        if self.points is not None:
            if len(self.points.shape) != 2:
                arcade.draw_point(self.points[0], self.points[1], arcade.color.BLACK, 3)
            else:
                for row in self.points:
                    arcade.draw_point(row[0], row[1], arcade.color.BLACK, 3)
        

    # Called whenever the mouse is pressed
    def on_mouse_press(self, x, y, button, modifiers):
        # New point
        new_point = np.array([x, y])

        # Append to list of points
        if self.points is None:
            self.points = new_point
        else:
            self.points = np.vstack((self.points, new_point))

        # Best fit circle  when the time is right
        if len(self.points) > 3:
            # Find X and Y depending on what shape we want
            if self.shape == 'c' or self.shape == 'circle':
                X = np.hstack((self.points, -np.ones(len(self.points)).reshape((-1,1))))
                y = -(self.points[:,0] ** 2 + self.points[:,1] ** 2)
            elif self.shape == 'l' or self.shape == 'line':
                X = np.hstack((self.points[:,0].reshape((-1, 1)), np.ones(len(self.points)).reshape((-1,1))))
                y = self.points[:,1]
            # perform regression
            for i, regressor in enumerate(self.regressors):
                self.regressors[i] = regressor.fit(X, y)


    # Called whenever a key is pressed
    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """

        # If the player presses a key, update the speed
        if key == arcade.key.SPACE:
            self.X = None
            self.y = None
            self.xc = 0
            self.yc = 0
            self.r = 0
            arcade.cleanup_texture_cache()
            arcade.set_background_color(arcade.color.SKY_BLUE)
            arcade.finish_render()

def main(shape, alpha):
    game = MyGame(shape, alpha)
    arcade.run()

if __name__ == "__main__":
    assert len(sys.argv) == 3
    main(sys.argv[1], float(sys.argv[2]))