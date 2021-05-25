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

    def __init__(self, shape):
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.LIGHT_GRAY)

        # List of points so far
        self.points = None

        # List of best parameters so far
        if shape == 'l' or shape == 'line':
            self.params = np.zeros(2)
        else:
            self.params = np.zeros(3)

        # Store shape of interest
        self.shape = shape

        # Store the regularization strengths
        self.alpha_l1 = 0.5
        self.alpha_l2 = 0.05

        max_iter = 1e7

        # Create regression objects for later
        self.regressors = [
            LinearRegression(fit_intercept=False), 
            Lasso(alpha=self.alpha_l1, fit_intercept=False, max_iter=max_iter), 
            Ridge(alpha=self.alpha_l2, fit_intercept=False, max_iter=max_iter)
        ]

        # Colors for reach type of regression
        self.colors = [arcade.color.RED, arcade.color.ARMY_GREEN, arcade.color.BLUE]

    
    def draw_stats(self):
        
        # First show Regularizers 
        arcade.draw_text(f'LASSO Regularization = {self.alpha_l1:.4f}', SCREEN_WIDTH // 7, 7 * SCREEN_HEIGHT // 8, self.colors[1])
        arcade.draw_text(f'Ridge Regularization = {self.alpha_l2:.4f}', SCREEN_WIDTH // 7, 7 * SCREEN_HEIGHT // 8 - 20, self.colors[2])

        # Draw legend @ top right
        x_legend = SCREEN_WIDTH - 200
        y_legend = SCREEN_HEIGHT - 100
        for col, reg in zip(self.colors, self.regressors):
            name = str(type(reg)).split(".")[-1][:-2]

            arcade.draw_text(name, x_legend, y_legend, col, 12)
            y_legend -= 0.02 * SCREEN_HEIGHT


    def on_draw(self):
        # This command has to happen before we start drawing
        arcade.start_render()

        # Show the numbers
        self.draw_stats()

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
                elif self.shape == 'p' or self.shape == 'parabola':
                    a, b, c = params
                    x_axis = np.arange(SCREEN_WIDTH).reshape((-1,1))
                    points = np.hstack((x_axis, a * x_axis ** 2 + b * x_axis + c))

                    # Draw best parabola so far
                    arcade.draw_lines(points, col, 1)
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
    
    # Helper for estimating next best curve
    def fit_regressors(self):
        # Best fit circle  when the time is right
        if len(self.points) > 3:
            # Find X and Y depending on what shape we want
            if self.shape == 'c' or self.shape == 'circle':
                X = np.hstack((self.points, -np.ones(len(self.points)).reshape((-1,1))))
                y = -(self.points[:,0] ** 2 + self.points[:,1] ** 2)
            elif self.shape == 'p' or self.shape == 'parabola':
                x_temp = self.points[:,0].reshape((-1,1))
                X = np.hstack((x_temp ** 2, x_temp, np.ones(len(x_temp)).reshape((-1,1))))
                y = self.points[:,1]
            elif self.shape == 'l' or self.shape == 'line':
                X = np.hstack((self.points[:,0].reshape((-1, 1)), np.ones(len(self.points)).reshape((-1,1))))
                y = self.points[:,1]
            # perform regression
            for i, regressor in enumerate(self.regressors):
                self.regressors[i] = regressor.fit(X, y)
        

    # Called whenever the mouse is pressed
    def on_mouse_press(self, x, y, button, modifiers):
        # New point
        new_point = np.array([x, y])

        # Append to list of points
        if self.points is None:
            self.points = new_point
        else:
            self.points = np.vstack((self.points, new_point))

        # Fit the curves
        self.fit_regressors()


    # Called whenever a key is pressed
    def on_key_press(self, key, modifiers):
        # If the player presses a key, update the speed
        if key == arcade.key.SPACE:
            self.points = None
            arcade.cleanup_texture_cache()
            arcade.set_background_color(arcade.color.LIGHT_GRAY)
            arcade.finish_render()
        elif key == arcade.key.LEFT or key == arcade.key.RIGHT or key == arcade.key.DOWN or key == arcade.key.UP:
            if key == arcade.key.LEFT:
                self.alpha_l1 *= 0.9
            elif key == arcade.key.RIGHT:
                self.alpha_l1 *= 1.1
            elif key == arcade.key.UP:
                self.alpha_l2 *= 1.1
            elif key == arcade.key.DOWN:
                self.alpha_l2 *= 0.9
            self.regressors[1].alpha = self.alpha_l1
            self.regressors[2].alpha = self.alpha_l2
            self.fit_regressors()
                

def main(shape):
    game = MyGame(shape)
    arcade.run()

if __name__ == "__main__":
    assert len(sys.argv) == 2
    main(sys.argv[1])