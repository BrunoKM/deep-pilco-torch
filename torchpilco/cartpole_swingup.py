"""
Cart pole swing-up: Identical version to PILCO V0.9
"""
import torch
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)


class CartPoleSwingUp(gym.Env):
    """
    Swing-up CartPole task as defined in:
        'Improving PILCO with Bayesian Neural Network Dynamics Models' by Yarin Gal et al.
        http://mlg.eng.cam.ac.uk/yarin/PDFs/DeepPILCO.pdf
    The state comprises [x_c, x_c_dot, theta, theta_dot] where x_c and theta are the cartpole
    horisontal position and pole angle respectively, and *_dot are their
    respective time derivatives.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # Define the physical constants and parameters
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.length = 0.6  # pole's length
        self.m_p_l = (self.m_p * self.length)
        self.force_mag = 10.0
        self.dt = 0.1  # seconds between state updates
        self.b = 0.1  # friction coefficient
        self.cost_sigma = 0.25  #  sigma_c from paper - length coefficient for cost

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # Distance of cart from origin at which to fail the episode
        # self.x_threshold = 2.4

        state_numerical_limit = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-self.force_mag,
                                       self.force_mag, shape=(1,))
        self.observation_space = spaces.Box(-state_numerical_limit, state_numerical_limit)

        self._seed()
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ Apply Euler's Difference algorithm to simulate a single step.
        """
        # Clip the action
        action = action.reshape([1])
        action = np.clip(action, -self.force_mag, self.force_mag)[0]  # TODO: why is this here?

        x, x_dot, theta, theta_dot = self.state

        s = math.sin(theta)
        c = math.cos(theta)

        # Find the time derivative the state
        xdot_update = (-2 * self.m_p_l * (theta_dot**2) * s
                       + 3 * self.m_p * self.g * s * c + 4 * action
                       - 4 * self.b * x_dot)/(4 * self.total_m - 3 * self.m_p * c**2)
        thetadot_update = ((-3 * self.m_p_l * (theta_dot**2) * s * c
                            + 6 * self.total_m * self.g * s
                            + 6 * (action - self.b * x_dot) * c)
                           / (4 * self.length * self.total_m - 3 * self.m_p_l * c**2))
        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt
        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt

        self.state = np.array((x, x_dot, theta, theta_dot))
        cost = self.compute_cost()
        return self.state, -cost, False, {}

    def compute_cost(self):
        cost = cartpole_cost_numpy(self.state, self.length, self.cost_sigma)
        return cost

    def reset(self):
        self.state = self.np_random.normal(loc=np.array(
            [0.0, 0.0, np.pi, 0.0]), scale=np.array([0.02, 0.02, 0.02, 0.02]))
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = 5  # max visible position of cart
        scale = screen_width/world_width
        carty = 200  # TOP OF CART
        polewidth = 6.0
        polelen = scale*self.length  # 0.6 or self.l
        cart_width = 40.0
        cart_height = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cart_width/2, cart_width/2, cart_height/2, -cart_height/2

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth/2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cart_height/4)
            self.wheel_r = rendering.make_circle(cart_height/4)
            self.wheeltrans_l = rendering.Transform(
                translation=(-cart_width/2, -cart_height/2))
            self.wheeltrans_r = rendering.Transform(
                translation=(cart_width/2, -cart_height/2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line(
                (0, carty - cart_height/2 - cart_height/4),
                (screen_width, carty - cart_height/2 - cart_height/4))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(-self.length *
                                            np.sin(x[2]), self.length*np.cos(x[2]))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        return


def cartpole_cost_numpy(state, pole_length, cost_sigma=0.25):
    cart_x = state[0]
    theta = state[2]
    pole_x = pole_length*np.sin(theta)
    pole_y = pole_length*np.cos(theta)
    tip_position = np.array([cart_x + pole_x, pole_y])

    target = np.array([0.0, pole_length])
    sq_distance = np.sum((tip_position - target)**2)
    cost = 1 - np.exp(-0.5*sq_distance/cost_sigma**2)
    return cost
    

def cartpole_cost_torch(states, pole_length=0.6, cost_sigma=0.25):
    cart_x = states[:, 0]
    theta = states[:, 2]
    pole_x = pole_length*torch.sin(theta)
    pole_y = pole_length*torch.cos(theta)
    tip_position = torch.stack([pole_x + cart_x, pole_y], dim=1)

    target = torch.tensor([0.0, pole_length], requires_grad=False)
    sq_distance = torch.sum((tip_position - target)**2, dim=1)
    cost = -torch.exp(-0.5*sq_distance/cost_sigma**2) + 1
    return cost
    

# def cost(states, sigma=0.25):
#     """Pendulum-v0: Same as OpenAI-Gym"""
#     l = 0.6
    
#     goal = Variable(torch.FloatTensor([0.0, l])).cuda()

#     # Cart position
#     cart_x = states[:, 0]
#     # Pole angle
#     thetas = states[:, 2]
#     # Pole position
#     x = torch.sin(thetas)*l
#     y = torch.cos(thetas)*l
#     positions = torch.stack([cart_x + x, y], 1)
    
#     squared_distance = torch.sum((goal - positions)**2, 1)
#     squared_sigma = sigma**2
#     cost = 1 - torch.exp(-0.5*squared_distance/squared_sigma)
    
#     return cost