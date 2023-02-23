import gymnasium as gym
from gymnasium import spaces
import pygame

import numpy as np
from collections import OrderedDict

from pixell import enmap, reproject
import healpy as hp

from . import schedlib


class SchedulerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self, 
        render_mode=None, 
        t0=0, 
        t1=1, 
        az0=90, 
        el0=45, 
        srate=1, 
        nside=128,
        target_geometry=None
    ):
        self.window = None
        self.clock = None
        self.window_size = 256

        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        if target_geometry is None:
            self.target_geometry = enmap.band_geometry(dec_cut=np.deg2rad([-60,30]), res=np.deg2rad(1))
        else:
            self.target_geometry = target_geometry
        shape = self.target_geometry[0]

        self._t0 = t0        # start time of the game
        self._t1 = t1        # end time of the game
        self._az0 = az0      # initial azimuth angle
        self._el0 = el0      # initial elevation angle
        self._srate = srate  # sampling rate

        self.observation_space = spaces.Dict(
            {
                "t": spaces.Box(t0, t1, shape=(1,)),            # current time: FIXME
                "az": spaces.Box(0, 360, shape=(1,)),           # current azimuth angle
                "el": spaces.Box(0, 90, shape=(1,)),            # current elevation angle
                "hitcount": spaces.Box(0, np.inf, shape=shape),     # hitcount: fractional hitcount from 0 to 1. 
                # "sun": spaces.Dict({
                #     "az": spaces.Box(0, 360, shape=(1,)),       # azimuth angle of the sun
                #     "el": spaces.Box(0, 90, shape=(1,)),        # elevation angle of the sun
                # })
            }
        )

        self.action_space = spaces.Dict(
            {
                "az": spaces.Box(0, 360, shape=(1,)),         # azimuth angle
                "el": spaces.Box(0, 90, shape=(1,)),          # elevation angle
                "throw": spaces.Box(0, 360, shape=(1,)),      # throw angle
                "on": spaces.MultiBinary(1),                  # whether data collection is on or off
                "velocity": spaces.Box(0, 1, shape=(1,)),     # speed of movement
                "duration": spaces.Box(0, 3600, shape=(1,)),  # duration of data collection: maximum duration for each action: 1 hours
            }
        )

        # time costs
        self._time_costs = {
            'move': 60  # sec
        }

        # current state
        self._state = OrderedDict({
            't': t0,
            'az': az0,
            'el': el0,
            'hitcount': np.zeros(self.npix),
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._state['t'] = self._t0
        self._state['az'] = self._az0
        self._state['el'] = self._el0
        self._state['hitcount'] = np.zeros(self.npix)

        if self.render_mode == "human":
            self._render_frame()
        return self._state, {}


    def step(self, action):
        # check if we need to move first
        if not (np.isclose(action['az'], self._state['az']) and np.isclose(action['el'], self._state['el'])):
            self._state['t'] += self._time_costs['move']
            self._state['az'] = action['az']
            self._state['el'] = action['el']
        # check whether we want to collect data
        if action['on'] == 0:
            # if we don't collect data, simply wait at the current location for the duration
            self._state['t'] += action['duration']
        else:
            # if we collect data, start scanning
            scan = schedlib.Scan(t0=self._state['t'], t1=self._state['t']+action['duration'], az=self._state['az'], el=self._state['el'], throw=action['throw'], velocity=action['velocity'])
            # update hitcount (in-place)
            schedlib.scan2hitcount(scan, hitcount=self._state['hitcount'])
            # update state
            ts, az, el = schedlib.get_azel(scan, srate=self._srate)
            self._state['az'] = az[-1]
            self._state['el'] = el[-1]
            self._state['t'] = ts[-1]
            
        observation = {
            'az': self._state['az'],
            'el': self._state['el'],
            't': self._state['t'],
            'hitcount': project_hitcount(self._state['hitcount'], self.target_geometry)
        }
        terminated = True if self._state['t'] > self._t1 else False
        info = {}
        reward = 1  # TODO:  implement this

        if self.render_mode == "human": self._render_frame()
        return observation, reward, terminated, info



    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                # (self.window_size, self.window_size)
                self.target_geometry[0][::-1]
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        hitcount = project_hitcount(self._state['hitcount'], self.target_geometry)
        hitcount_surface = pygame.surfarray.make_surface(hitcount)
        # canvas = pygame.Surface((self.window_size, self.window_size))
        canvas = pygame.Surface(self.target_geometry[0][::-1])
        canvas.fill((255, 255, 255))

        if self.render_mode == "human":
            # display the hitcount surface on the canvas
            canvas.blit(hitcount_surface, (0, 0))

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            # draw an image of the sky
            # plot the array of hitcount in pygame
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# utility functions
def project_hitcount(hitcount_hp, car_geometry):
    # project healpix to CAR pixelization
    hitcount_car = reproject.enmap_from_healpix(hitcount_hp, *car_geometry) 
    return np.array(hitcount_car[0])