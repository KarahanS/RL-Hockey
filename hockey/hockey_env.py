import math
from typing import Any, Dict, Tuple
import numpy as np

import Box2D

# noinspection PyUnresolvedReferences
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding, EzPickle
from enum import Enum

# import pyglet
# from pyglet import gl

FPS = 50
SCALE = 60.0  # affects how fast-paced the game is, forces should be adjusted as well (Don't touch)

VIEWPORT_W = 600
VIEWPORT_H = 480
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W / 2
CENTER_Y = H / 2
ZONE = W / 20
MAX_ANGLE = math.pi / 3  # Maximimal angle of racket
MAX_TIME_KEEP_PUCK = 15
GOAL_SIZE = 75

RACKETPOLY = [
    (-10, 20),
    (+5, 20),
    (+5, -20),
    (-10, -20),
    (-18, -10),
    (-21, 0),
    (-18, 10),
]
RACKETFACTOR = 1.2

FORCEMULTIPLIER = 6000
SHOOTFORCEMULTIPLIER = 60
TORQUEMULTIPLIER = 400
MAX_PUCK_SPEED = 25


def dist_positions(p1, p2):
    return np.sqrt(np.sum(np.asarray(p1 - p2) ** 2, axis=-1))


class ContactDetector(contactListener):
    def __init__(self, env, verbose=False):
        contactListener.__init__(self)
        self.env = env
        self.verbose = verbose

    def BeginContact(self, contact):
        if (
            self.env.goal_player_2 == contact.fixtureA.body
            or self.env.goal_player_2 == contact.fixtureB.body
        ):
            if (
                self.env.puck == contact.fixtureA.body
                or self.env.puck == contact.fixtureB.body
            ):
                if self.verbose:
                    print("Player 1 scored")
                self.env.done = True
                self.env.winner = 1
        if (
            self.env.goal_player_1 == contact.fixtureA.body
            or self.env.goal_player_1 == contact.fixtureB.body
        ):
            if (
                self.env.puck == contact.fixtureA.body
                or self.env.puck == contact.fixtureB.body
            ):
                if self.verbose:
                    print("Player 2 scored")
                self.env.done = True
                self.env.winner = -1
        if (
            contact.fixtureA.body == self.env.player1
            or contact.fixtureB.body == self.env.player1
        ) and (
            contact.fixtureA.body == self.env.puck
            or contact.fixtureB.body == self.env.puck
        ):
            if self.env.keep_mode and self.env.puck.linearVelocity[0] < 0.1:
                if self.env.player1_has_puck == 0:
                    self.env.player1_has_puck = MAX_TIME_KEEP_PUCK

        if (
            contact.fixtureA.body == self.env.player2
            or contact.fixtureB.body == self.env.player2
        ) and (
            contact.fixtureA.body == self.env.puck
            or contact.fixtureB.body == self.env.puck
        ):
            if self.env.keep_mode and self.env.puck.linearVelocity[0] > -0.1:
                if self.env.player2_has_puck == 0:
                    self.env.player2_has_puck = MAX_TIME_KEEP_PUCK

    def EndContact(self, contact):
        pass


class Mode(Enum):
    NORMAL = 0
    TRAIN_SHOOTING = 1
    TRAIN_DEFENSE = 2


class HockeyEnv(gym.Env, EzPickle):
    """
    Observation Space:
        - 0  x pos player one
        - 1  y pos player one
        - 2  angle player one
        - 3  x vel player one
        - 4  y vel player one
        - 5  angular vel player one
        - 6  x player two
        - 7  y player two
        - 8  angle player two
        - 9  y vel player two
        - 10 y vel player two
        - 11 angular vel player two
        - 12 x pos puck
        - 13 y pos puck
        - 14 x vel puck
        - 15 y vel puck
        - Keep Puck Mode
        - 16 time left player has puck
        - 17 time left other player has puck

    Action Space (Discrete(8)):
        - Action 0: do nothing
        - Action 1: -1 in x
        - Action 2: 1 in x
        - Action 3: -1 in y
        - Action 4: 1 in y
        - Action 5: -1 in angle
        - Action 6: 1 in angle
        - Action 7: shoot (if keep_mode is on)
    """

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": FPS}

    continuous = False

    def __init__(
        self,
        keep_mode: bool = True,
        mode: int | str | Mode = Mode.NORMAL,
        verbose: bool = False,
    ):
        """
        Build and environment instance

        Args:
          keep_mode (bool, optional): whether the puck gets catched by the player.
              This can be changed later using the reset function. Defaults to True.
          mode (int | str | Mode, optional): mode: is the game mode: NORMAL (0),
              TRAIN_SHOOTING (1), TRAIN_DEFENSE (2). Defaults to Mode.NORMAL.
          verbose (bool, optional): Verbose logging. Defaults to False.
        """
        EzPickle.__init__(self)
        self.set_seed()
        self.screen = None
        self.clock = None
        self.surf = None
        self.isopen = True
        self.mode = mode
        self.keep_mode = keep_mode
        self.player1_has_puck = 0
        self.player2_has_puck = 0

        self.world = Box2D.b2World([0, 0])
        self.player1 = None
        self.player2 = None
        self.puck = None
        self.goal_player_1 = None
        self.goal_player_2 = None
        self.world_objects = []
        self.drawlist = []
        self.done = False
        self.winner = 0
        self.one_starts = True  # player one starts the game (alternating)

        self.timeStep = 1.0 / FPS
        self.time = 0
        self.max_timesteps = None  # see reset

        self.closest_to_goal_dist = 1000

        # 0  x pos player one
        # 1  y pos player one
        # 2  angle player one
        # 3  x vel player one
        # 4  y vel player one
        # 5  angular vel player one
        # 6  x player two
        # 7  y player two
        # 8  angle player two
        # 9  y vel player two
        # 10 y vel player two
        # 11 angular vel player two
        # 12 x pos puck
        # 13 y pos puck
        # 14 x vel puck
        # 15 y vel puck
        # Keep Puck Mode
        # 16 time left player has puck
        # 17 time left other player has puck
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(18,), dtype=np.float32
        )

        # linear force in (x,y)-direction and torque
        self.num_actions = 3 if not self.keep_mode else 4
        self.action_space = spaces.Box(
            -1, +1, (self.num_actions * 2,), dtype=np.float32
        )

        # see discrete_to_continous_action()
        self.discrete_action_space = spaces.Discrete(7)

        self.verbose = verbose

        self.reset(self.one_starts)

    def set_seed(self, seed: int = None):
        """set seed. If no argument provided or seed=None. Set a random seed

        Args:
          seed (int, optional): seed. Defaults to None.

        Returns:
          List[int]: in list embedded seed
        """
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed
        return [seed]

    def _destroy(self):
        if self.player1 is None:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.player1)
        self.player1 = None
        self.world.DestroyBody(self.player2)
        self.player2 = None
        self.world.DestroyBody(self.puck)
        self.puck = None
        self.world.DestroyBody(self.goal_player_1)
        self.goal_player_1 = None
        self.world.DestroyBody(self.goal_player_2)
        self.goal_player_2 = None
        for obj in self.world_objects:
            self.world.DestroyBody(obj)
        self.world_objects = []
        self.drawlist = []

    def r_uniform(self, mini, maxi):
        return self.np_random.uniform(mini, maxi, 1)[0]

    def _create_player(self, position, color, is_player_two):
        player = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[
                        (
                            (
                                -x / SCALE * RACKETFACTOR
                                if is_player_two
                                else x / SCALE * RACKETFACTOR
                            ),
                            y / SCALE * RACKETFACTOR,
                        )
                        for x, y in RACKETPOLY
                    ]
                ),
                density=200.0 / RACKETFACTOR,
                friction=1.0,
                categoryBits=0x0010,
                maskBits=0x011,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        player.color1 = color
        player.color2 = color
        # player.linearDamping = 0.1
        player.anguarDamping = 1.0

        return player

    def _create_puck(self, position, color):
        puck = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=13 / SCALE, pos=(0, 0)),
                density=7.0,
                friction=0.1,
                categoryBits=0x001,
                maskBits=0x0010,
                restitution=0.95,
            ),  # 0.99 bouncy
        )
        puck.color1 = color
        puck.color2 = color
        puck.linearDamping = 0.05

        return puck

    def _create_world(self):
        def _create_wall(position, poly):
            wall = self.world.CreateStaticBody(
                position=position,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x / SCALE, y / SCALE) for x, y in poly]
                    ),
                    density=0,
                    friction=0.1,
                    categoryBits=0x011,
                    maskBits=0x0011,
                ),
            )
            wall.color1 = (0, 0, 0)
            wall.color2 = (0, 0, 0)

            return wall

        def _create_decoration():
            objs = []
            objs.append(
                self.world.CreateStaticBody(
                    position=(W / 2, H / 2),
                    angle=0.0,
                    fixtures=fixtureDef(
                        shape=circleShape(radius=100 / SCALE, pos=(0, 0)),
                        categoryBits=0x0,
                        maskBits=0x0,
                    ),
                )
            )
            objs[-1].color1 = (204, 204, 204)
            objs[-1].color2 = (204, 204, 204)

            # left goal
            objs.append(
                self.world.CreateStaticBody(
                    position=(W / 2 - 250 / SCALE, H / 2),
                    angle=0.0,
                    fixtures=fixtureDef(
                        shape=circleShape(radius=GOAL_SIZE / SCALE, pos=(0, 0)),
                        categoryBits=0x0,
                        maskBits=0x0,
                    ),
                )
            )
            orange = (239, 203, 138)
            objs[-1].color1 = orange
            objs[-1].color2 = orange

            poly = [(0, 100), (100, 100), (100, -100), (0, -100)]
            objs.append(
                self.world.CreateStaticBody(
                    position=(W / 2 - 240 / SCALE, H / 2),
                    angle=0.0,
                    fixtures=fixtureDef(
                        shape=polygonShape(
                            vertices=[(x / SCALE, y / SCALE) for x, y in poly]
                        ),
                        categoryBits=0x0,
                        maskBits=0x0,
                    ),
                )
            )
            objs[-1].color1 = (255, 255, 255)
            objs[-1].color2 = (255, 255, 255)

            # right goal
            objs.append(
                self.world.CreateStaticBody(
                    position=(W / 2 + 250 / SCALE, H / 2),
                    angle=0.0,
                    fixtures=fixtureDef(
                        shape=circleShape(radius=GOAL_SIZE / SCALE, pos=(0, 0)),
                        categoryBits=0x0,
                        maskBits=0x0,
                    ),
                )
            )
            objs[-1].color1 = orange
            objs[-1].color2 = orange

            poly = [(100, 100), (0, 100), (0, -100), (100, -100)]
            objs.append(
                self.world.CreateStaticBody(
                    position=(W / 2 + 140 / SCALE, H / 2),
                    angle=0.0,
                    fixtures=fixtureDef(
                        shape=polygonShape(
                            vertices=[(x / SCALE, y / SCALE) for x, y in poly]
                        ),
                        categoryBits=0x0,
                        maskBits=0x0,
                    ),
                )
            )
            objs[-1].color1 = (255, 255, 255)
            objs[-1].color2 = (255, 255, 255)

            return objs

        self.world_objects = []

        self.world_objects.extend(_create_decoration())

        poly = [(-250, 10), (-250, -10), (250, -10), (250, 10)]
        self.world_objects.append(_create_wall((W / 2, H - 0.5), poly))
        self.world_objects.append(_create_wall((W / 2, 0.5), poly))

        poly = [
            (-10, (H - 1) / 2 * SCALE - GOAL_SIZE),
            (10, (H - 1) / 2 * SCALE - GOAL_SIZE - 7),
            (10, -5),
            (-10, -5),
        ]
        self.world_objects.append(
            _create_wall((W / 2 - 245 / SCALE, H - 0.5), [(x, -y) for x, y in poly])
        )
        self.world_objects.append(_create_wall((W / 2 - 245 / SCALE, 0.5), poly))

        self.world_objects.append(
            _create_wall((W / 2 + 245 / SCALE, H - 0.5), [(-x, -y) for x, y in poly])
        )
        self.world_objects.append(
            _create_wall((W / 2 + 245 / SCALE, 0.5), [(-x, y) for x, y in poly])
        )

        self.drawlist.extend(self.world_objects)

    def _create_goal(self, position, poly):
        goal = self.world.CreateStaticBody(
            position=position,
            angle=0.0,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x / SCALE, y / SCALE) for x, y in poly]
                    ),
                    density=0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    isSensor=True,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x / SCALE, y / SCALE) for x, y in poly]
                    ),
                    density=0,
                    friction=0.1,
                    categoryBits=0x010,
                    maskBits=0x0010,
                ),
            ],
        )
        goal.color1 = (128, 128, 128)
        goal.color2 = (128, 128, 128)

        return goal

    def reset(
        self,
        one_starting: bool = None,
        mode: str | int | Mode = None,
        seed: int = None,
        options: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """reset the environment

          Args:
            one_starting (bool, optional): player one starts. Defaults to None.
            mode (str | int | Mode, optional): environment mode analog to `mode` in __init__. Defaults to None.
            seed (int, optional): random seed for env. Defaults to None.
            options (Dict[str, Any], optional): Not used. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: observation, info dictionary
        """
        self._destroy()
        self.set_seed(seed)
        self.world.contactListener_keepref = ContactDetector(self, verbose=self.verbose)
        self.world.contactListener = self.world.contactListener_keepref
        self.done = False
        self.winner = 0
        self.prev_shaping = None
        self.time = 0
        if mode is not None and hasattr(Mode, str(self.mode)):
            self.mode = mode

        if self.mode == Mode.NORMAL:
            self.max_timesteps = 250
            if one_starting is not None:
                self.one_starts = one_starting
            else:
                self.one_starts = not self.one_starts
        else:
            self.max_timesteps = 80
        self.closest_to_goal_dist = 1000

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create world
        self._create_world()

        poly = [(-10, GOAL_SIZE), (10, GOAL_SIZE), (10, -GOAL_SIZE), (-10, -GOAL_SIZE)]
        self.goal_player_1 = self._create_goal(
            (W / 2 - 245 / SCALE - 10 / SCALE, H / 2), poly
        )
        self.goal_player_2 = self._create_goal(
            (W / 2 + 245 / SCALE + 10 / SCALE, H / 2), poly
        )

        # Create players
        red = (235, 98, 53)
        self.player1 = self._create_player((W / 5, H / 2), red, False)
        blue = (93, 158, 199)
        if self.mode != Mode.NORMAL:
            self.player2 = self._create_player(
                (
                    4 * W / 5 + self.r_uniform(-W / 3, W / 6),
                    H / 2 + self.r_uniform(-H / 4, H / 4),
                ),
                blue,
                True,
            )
        else:
            self.player2 = self._create_player((4 * W / 5, H / 2), blue, True)
        if self.mode == Mode.NORMAL or self.mode == Mode.TRAIN_SHOOTING:
            if self.one_starts or self.mode == Mode.TRAIN_SHOOTING:
                self.puck = self._create_puck(
                    (
                        W / 2 - self.r_uniform(H / 8, H / 4),
                        H / 2 + self.r_uniform(-H / 8, H / 8),
                    ),
                    (0, 0, 0),
                )
            else:
                self.puck = self._create_puck(
                    (
                        W / 2 + self.r_uniform(H / 8, H / 4),
                        H / 2 + self.r_uniform(-H / 8, H / 8),
                    ),
                    (0, 0, 0),
                )
        elif self.mode == Mode.TRAIN_DEFENSE:
            self.puck = self._create_puck(
                (
                    W / 2 + self.r_uniform(0, W / 3),
                    H / 2 + 0.8 * self.r_uniform(-H / 2, H / 2),
                ),
                (0, 0, 0),
            )
            direction = self.puck.position - (
                0,
                H / 2 + 0.6 * self.r_uniform(-GOAL_SIZE / SCALE, GOAL_SIZE / SCALE),
            )
            direction = direction / direction.length
            force = -direction * SHOOTFORCEMULTIPLIER * self.puck.mass / self.timeStep
            self.puck.ApplyForceToCenter(force, True)
        # Todo get the scaling right

        self.drawlist.extend([self.player1, self.player2, self.puck])

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _check_boundaries(self, force, player, is_player_one):
        if (
            (
                is_player_one
                and player.position[0] < W / 2 - 210 / SCALE
                and force[0] < 0
            )
            or (
                not is_player_one
                and player.position[0] > W / 2 + 210 / SCALE
                and force[0] > 0
            )
            or (is_player_one and player.position[0] > W / 2 and force[0] > 0)
            or (not is_player_one and player.position[0] < W / 2 and force[0] < 0)
        ):  # Do not leave playing area to the left/right
            vel = player.linearVelocity
            player.linearVelocity[0] = 0
            force[0] = -vel[0]
        if (player.position[1] > H - 1.2 and force[1] > 0) or (
            player.position[1] < 1.2 and force[1] < 0
        ):  # Do not leave playing area to the top/bottom
            vel = player.linearVelocity
            player.linearVelocity[1] = 0
            force[1] = -vel[1]

        return np.asarray(force, dtype=float)

    def _apply_translation_action_with_max_speed(
        self, player, action, max_speed, is_player_one
    ):
        velocity = np.asarray(player.linearVelocity)
        speed = np.sqrt(np.sum((velocity) ** 2))
        if is_player_one:
            force = action * FORCEMULTIPLIER
        else:
            force = -action * FORCEMULTIPLIER

        if (is_player_one and player.position[0] > CENTER_X - ZONE) or (
            not is_player_one and player.position[0] < CENTER_X + ZONE
        ):  # bounce at the center line
            force[0] = 0
            if is_player_one:
                if player.linearVelocity[0] > 0:
                    force[0] = (
                        -2 * player.linearVelocity[0] * player.mass / self.timeStep
                    )
                force[0] += (
                    -1
                    * (player.position[0] - CENTER_X)
                    * player.linearVelocity[0]
                    * player.mass
                    / self.timeStep
                )
            else:
                if player.linearVelocity[0] < 0:
                    force[0] = (
                        -2 * player.linearVelocity[0] * player.mass / self.timeStep
                    )
                force[0] += (
                    1
                    * (player.position[0] - CENTER_X)
                    * player.linearVelocity[0]
                    * player.mass
                    / self.timeStep
                )

            player.linearDamping = 20.0

            player.ApplyForceToCenter(
                self._check_boundaries(force, player, is_player_one).tolist(), True
            )
            return

        if speed < max_speed:
            player.linearDamping = 5.0
            player.ApplyForceToCenter(
                self._check_boundaries(force.tolist(), player, is_player_one), True
            )
        else:
            player.linearDamping = 20.0
            deltaVelocity = self.timeStep * force / player.mass
            if np.sqrt(np.sum((velocity + deltaVelocity) ** 2)) < speed:
                player.ApplyForceToCenter(
                    self._check_boundaries(force.tolist(), player, is_player_one), True
                )
            else:
                pass

    def _apply_rotation_action_with_max_speed(self, player, action):
        angle = np.asarray(player.angle)
        torque = action * TORQUEMULTIPLIER
        if abs(angle) > MAX_ANGLE:  # limit rotation
            torque = 0
            if player.angle * player.angularVelocity > 0:
                torque = -0.1 * player.angularVelocity * player.mass / self.timeStep
            torque += -0.1 * (player.angle) * player.mass / self.timeStep
            player.angularDamping = 10.0
        else:
            player.angularDamping = 2.0
        player.ApplyTorque(float(torque), True)

    def _get_obs(self):
        obs = np.hstack(
            [
                self.player1.position - [CENTER_X, CENTER_Y],
                [self.player1.angle],
                self.player1.linearVelocity,
                [self.player1.angularVelocity],
                self.player2.position - [CENTER_X, CENTER_Y],
                [self.player2.angle],
                self.player2.linearVelocity,
                [self.player2.angularVelocity],
                self.puck.position - [CENTER_X, CENTER_Y],
                self.puck.linearVelocity,
            ]
            + (
                []
                if not self.keep_mode
                else [self.player1_has_puck, self.player2_has_puck]
            )
        )
        return obs

    def obs_agent_two(self):
        """returns the observations for agent two (symmetric mirrored version of agent one)

        Returns:
            np.ndarray: observation array for agent
        """
        obs = np.hstack(
            [
                -(self.player2.position - [CENTER_X, CENTER_Y]),
                [self.player2.angle],  # the angle is already rotationally symmetric
                -self.player2.linearVelocity,
                [self.player2.angularVelocity],
                -(self.player1.position - [CENTER_X, CENTER_Y]),
                [self.player1.angle],  # the angle is already rotationally symmetric
                -self.player1.linearVelocity,
                [self.player1.angularVelocity],
                -(self.puck.position - [CENTER_X, CENTER_Y]),
                -self.puck.linearVelocity,
            ]
            + (
                []
                if not self.keep_mode
                else [self.player2_has_puck, self.player1_has_puck]
            )
        )

        return obs

    def _compute_reward(self):
        r = 0

        if self.done:
            if self.winner == 0:  # tie
                r += 0
            elif self.winner == 1:  # you won
                r += 10
            else:  # opponent won
                r -= 10
        return float(r)

    def get_reward(self, info: Dict[str, Any]):
        """extract reward from info dict
        function that computes the reward returned to the agent, here with some reward shaping
        the shaping should probably be removed in future versions

        Args:
            info (Dict[str, Any]): info dict with key: "reward_closeness_to_puck"

        Returns:
            float: reward signal
        """
        r = self._compute_reward()
        r += info["reward_closeness_to_puck"]
        return float(r)

    def get_reward_agent_two(self, info_two: Dict[str, Any]):
        """extract reward from info dict for agent two

        Args:
            info (Dict[str, Any]): info dict with key: "reward_closeness_to_puck"

        Returns:
            float: reward signal
        """
        r = -self._compute_reward()
        r += info_two["reward_closeness_to_puck"]
        return float(r)

    def _get_info(self):
        # different proxy rewards:
        # Proxy reward/penalty for not being close to puck in the own half when puck is flying towards goal (not to opponent)
        reward_closeness_to_puck = 0
        if self.puck.position[0] < CENTER_X and self.puck.linearVelocity[0] <= 0:
            dist_to_puck = dist_positions(self.player1.position, self.puck.position)
            max_dist = 250.0 / SCALE
            max_reward = -30.0  # max (negative) reward through this proxy
            factor = max_reward / (max_dist * self.max_timesteps / 2)
            reward_closeness_to_puck += (
                dist_to_puck * factor
            )  # Proxy reward for being close to puck in the own half
        # Proxy reward: touch puck
        reward_touch_puck = 0.0
        if self.player1_has_puck == MAX_TIME_KEEP_PUCK:
            reward_touch_puck = 1.0

        # puck is flying in the right direction
        max_reward = 1.0
        factor = max_reward / (self.max_timesteps * MAX_PUCK_SPEED)
        reward_puck_direction = (
            self.puck.linearVelocity[0] * factor
        )  # Puck flies right is good and left not

        return {
            "winner": self.winner,
            "reward_closeness_to_puck": float(reward_closeness_to_puck),
            "reward_touch_puck": float(reward_touch_puck),
            "reward_puck_direction": float(reward_puck_direction),
        }

    def get_info_agent_two(self):
        """mirrored version of get_info_agent_one
        see get_info for player 1. here everything is just mirrored

        Returns:
            Dict[str, float]: info dictionary
        """
        reward_closeness_to_puck = 0
        if self.puck.position[0] > CENTER_X and self.puck.linearVelocity[0] >= 0:
            dist_to_puck = dist_positions(self.player2.position, self.puck.position)
            max_dist = 250.0 / SCALE
            max_reward = -30.0  # max (negative) reward through this proxy
            factor = max_reward / (max_dist * self.max_timesteps / 2)
            reward_closeness_to_puck += (
                dist_to_puck * factor
            )  # Proxy reward for being close to puck in the own half
        # Proxy reward: touch puck
        reward_touch_puck = 0.0
        if self.player2_has_puck == MAX_TIME_KEEP_PUCK:
            reward_touch_puck = 1.0

        # puck is flying in the left direction
        max_reward = 1.0
        factor = -max_reward / (self.max_timesteps * MAX_PUCK_SPEED)
        reward_puck_direction = (
            self.puck.linearVelocity[0] * factor
        )  # Puck flies left is good and right not

        return {
            "winner": -self.winner,
            "reward_closeness_to_puck": float(reward_closeness_to_puck),
            "reward_touch_puck": float(reward_touch_puck),
            "reward_puck_direction": float(reward_puck_direction),
        }

    def set_state(self, state: np.ndarray):
        """function to revert the state of the environment to a previous state (observation)

        Args:
            state (np.ndarray): return to this state
        """
        self.player1.position = (state[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
        self.player1.angle = math.atan2(state[2], state[3])
        self.player1.linearVelocity = [state[4], state[5]]
        self.player1.angularVelocity = state[6]
        self.player2.position = (state[[7, 8]] + [CENTER_X, CENTER_Y]).tolist()
        self.player2.angle = math.atan2(state[9], state[10])
        self.player2.linearVelocity = [state[11], state[12]]
        self.player2.angularVelocity = state[13]
        self.puck.position = (state[[14, 15]] + [CENTER_X, CENTER_Y]).tolist()
        self.puck.linearVelocity = [state[16], state[17]]

    def _limit_puck_speed(self):
        puck_speed = np.sqrt(
            self.puck.linearVelocity[0] ** 2 + self.puck.linearVelocity[1] ** 2
        )
        if puck_speed > MAX_PUCK_SPEED:
            self.puck.linearDamping = 10.0
        else:
            self.puck.linearDamping = 0.05
        self.puck.angularSpeed = 0

    def _keep_puck(self, player):
        self.puck.position = player.position
        self.puck.linearVelocity = player.linearVelocity

    def _shoot(self, player, is_player_one):
        # self.puck.position = player.position
        if is_player_one:
            self.puck.ApplyForceToCenter(
                Box2D.b2Vec2(math.cos(player.angle) * 1.0, math.sin(player.angle) * 1.0)
                * self.puck.mass
                / self.timeStep
                * SHOOTFORCEMULTIPLIER,
                True,
            )
        else:
            self.puck.ApplyForceToCenter(
                Box2D.b2Vec2(
                    math.cos(player.angle) * -1.0, math.sin(player.angle) * -1.0
                )
                * self.puck.mass
                / self.timeStep
                * SHOOTFORCEMULTIPLIER,
                True,
            )

    def discrete_to_continous_action(self, discrete_action: int) -> np.ndarray:
        """converts discrete actions into continuous ones (for each player)
        The actions allow only one operation each timestep, e.g. X or Y or angle change.

        This is surely limiting. Other discrete actions are possible
          - Action 0: do nothing
          - Action 1: -1 in x
          - Action 2: 1 in x
          - Action 3: -1 in y
          - Action 4: 1 in y
          - Action 5: -1 in angle
          - Action 6: 1 in angle
          - Action 7: shoot (if keep_mode is on)


        Args:
          discrete_action (int): action to choose from [0, ..., 7]

        Returns:
          np.ndarray: continuous action
            - x action
            - y action
            - angle to turn
            - if keep_dim == True: shoot or not
        """
        action_cont = [
            (discrete_action == 1) * -1.0 + (discrete_action == 2) * 1.0,  # player x
            (discrete_action == 3) * -1.0 + (discrete_action == 4) * 1.0,  # player y
            (discrete_action == 5) * -1.0 + (discrete_action == 6) * 1.0,
        ]  # player angle
        if self.keep_mode:
            action_cont.append((discrete_action == 7) * 1.0)

        return action_cont

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """apply action to environment

        Args:
          action (np.ndarray): either discrete or continuous action. The vector includes actions for agent 1 an agent 2.
            The first half is for agent 1, the second half is for agent 2.
            In case of continuous actions:
            - dim 0: translation in x
            - dim 1: translation in y
            - dim 2: rotation action
            - dim (3: only in keep mode -> shoot puck)
            - dim 4-7 same as for 0-3 but for agent 2
            In case of discrete actions (NOTE: currently not supported):
            - dim 0: discrete action for agent 1
            - dim 1: discrete action for agent 2

        Returns:
          Tuple[np.ndarray, float, bool, bool, dict[str, Any]]: observation, reward, done, truncated, info
        """
        action = np.clip(action, -1, +1).astype(np.float32)

        self._apply_translation_action_with_max_speed(
            self.player1, action[:2], 10, True
        )
        self._apply_rotation_action_with_max_speed(self.player1, action[2])
        player2_idx = 3 if not self.keep_mode else 4
        self._apply_translation_action_with_max_speed(
            self.player2, action[player2_idx : player2_idx + 2], 10, False
        )
        self._apply_rotation_action_with_max_speed(
            self.player2, action[player2_idx + 2]
        )

        self._limit_puck_speed()
        if self.keep_mode:
            if self.player1_has_puck > 1:
                self._keep_puck(self.player1)
                self.player1_has_puck -= 1
                if self.player1_has_puck == 1 or action[3] > 0.5:  # shooting
                    self._shoot(self.player1, True)
                    self.player1_has_puck = 0
            if self.player2_has_puck > 1:
                self._keep_puck(self.player2)
                self.player2_has_puck -= 1
                if (
                    self.player2_has_puck == 1 or action[player2_idx + 3] > 0.5
                ):  # shooting
                    self._shoot(self.player2, False)
                    self.player2_has_puck = 0

        self.world.Step(self.timeStep, 6 * 30, 2 * 30)

        obs = self._get_obs()
        if self.time >= self.max_timesteps:
            self.done = True

        info = self._get_info()
        reward = self.get_reward(info)

        self.closest_to_goal_dist = min(
            self.closest_to_goal_dist, dist_positions(self.puck.position, (W, H / 2))
        )
        self.time += 1
        # Todo: maybe use the truncation flag when the time runs out!
        return obs, reward, self.done, False, info

    def render(self, mode: str = "human") -> None | np.ndarray:
        """render the state of the environment

        Args:
          mode (str, optional): render mode. Defaults to "human".

        Returns:
          None | np.ndarray: depending on the render mode there is a return or not
        """
        if mode is None:
            gym.logger.warn("the render method needs a rendering mode")
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        radius=f.shape.radius * SCALE,
                        width=0,
                        center=trans * f.shape.pos * SCALE,
                        color=obj.color1,
                    )
                    pygame.draw.circle(
                        self.surf,
                        radius=f.shape.radius * SCALE,
                        width=2,
                        center=trans * f.shape.pos * SCALE,
                        color=obj.color2,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(
                        self.surf, points=path, color=obj.color1, width=0
                    )
                    path.append(path[0])
                    pygame.draw.polygon(
                        self.surf, points=path, color=obj.color2, width=2
                    )

        # self.score_label.draw()
        self.surf = pygame.transform.flip(self.surf, False, True)

        if mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        """
        close the environment after training
        """
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    @property
    def mode(self) -> Mode:
        return self._mode

    @mode.setter
    def mode(self, value: str | int | Mode):
        """
        Set the Enum object using an Enum, name, or value.
        """
        if isinstance(value, Mode):
            # If the input is already an Enum member, set it directly
            self._mode = value
        elif isinstance(value, str):
            # If the input is a name, convert it to the Enum
            try:
                self._mode = Mode[value]
            except KeyError:
                raise ValueError(f"{value} is not a valid name for {Mode.__name__}")
        elif isinstance(value, int):
            # If the input is a value, convert it to the Enum
            try:
                self._mode = Mode(value)
            except ValueError:
                raise ValueError(f"{value} is not a valid value for {Mode.__name__}")
        else:
            raise TypeError("Input value must be an Enum, name (str), or value (int)")


class BasicOpponent:
    def __init__(self, weak=True, keep_mode=True):
        self.weak = weak
        self.keep_mode = keep_mode
        self.phase = np.random.uniform(0, np.pi)

    def act(self, obs, verbose=False):
        alpha = obs[2]
        p1 = np.asarray([obs[0], obs[1], alpha])
        v1 = np.asarray(obs[3:6])
        puck = np.asarray(obs[12:14])
        puckv = np.asarray(obs[14:16])
        # print(p1,v1,puck,puckv)
        target_pos = p1[0:2]
        target_angle = p1[2]
        self.phase += np.random.uniform(0, 0.2)

        time_to_break = 0.1
        if self.weak:
            kp = 0.5
        else:
            kp = 10
        kd = 0.5

        # if ball flies towards our goal or very slowly away: try to catch it
        if puckv[0] < 30.0 / SCALE:
            dist = np.sqrt(np.sum((p1[0:2] - puck) ** 2))
            # Am I behind the ball?
            if p1[0] < puck[0] and abs(p1[1] - puck[1]) < 30.0 / SCALE:
                # Go and kick
                target_pos = [puck[0] + 0.2, puck[1] + puckv[1] * dist * 0.1]
            else:
                # get behind the ball first
                target_pos = [-210 / SCALE, puck[1]]
        else:  # go in front of the goal
            target_pos = [-210 / SCALE, 0]
        target_angle = MAX_ANGLE * np.sin(self.phase)
        shoot = 0.0
        if self.keep_mode and obs[16] > 0 and obs[16] < 7:
            shoot = 1.0

        target = np.asarray([target_pos[0], target_pos[1], target_angle])
        # use PD control to get to target
        error = target - p1
        need_break = abs((error / (v1 + 0.01))) < [
            time_to_break,
            time_to_break,
            time_to_break * 10,
        ]
        if verbose:
            print(error, abs(error / (v1 + 0.01)), need_break)

        action = np.clip(
            error * [kp, kp / 5, kp / 2] - v1 * need_break * [kd, kd, kd], -1, 1
        )
        if self.keep_mode:
            return np.hstack([action, [shoot]])
        else:
            return action


class HumanOpponent:
    def __init__(self, env, player=1):
        import pygame

        self.env = env
        self.player = player
        self.a = 0

        if env.screen is None:
            env.render()

        self.key_action_mapping = {
            pygame.K_LEFT: 1 if self.player == 1 else 2,  # Left arrow key
            pygame.K_UP: 4 if self.player == 1 else 3,  # Up arrow key
            pygame.K_RIGHT: 2 if self.player == 1 else 1,  # Right arrow key
            pygame.K_DOWN: 3 if self.player == 1 else 4,  # Down arrow key
            pygame.K_w: 5,  # w
            pygame.K_s: 6,  # s
            pygame.K_SPACE: 7,  # space
        }

        print("Human Controls:")
        print(" left:\t\t\tleft arrow key left")
        print(" right:\t\t\tarrow key right")
        print(" up:\t\t\tarrow key up")
        print(" down:\t\t\tarrow key down")
        print(" tilt clockwise:\tw")
        print(" tilt anti-clockwise:\ts")
        print(" shoot :\tspace")

    def act(self, obs):
        import pygame

        keys = pygame.key.get_pressed()
        action = 0
        for key in self.key_action_mapping.keys():
            if keys[key]:
                action = self.key_action_mapping[key]
        return self.env.discrete_to_continous_action(action)


class HockeyEnv_BasicOpponent(HockeyEnv):
    def __init__(self, mode=Mode.NORMAL, weak_opponent=False):
        super().__init__(mode=mode, keep_mode=True)
        self.opponent = BasicOpponent(weak=weak_opponent)
        # linear force in (x,y)-direction, torque, and shooting
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

    def step(self, action):
        ob2 = self.obs_agent_two()
        a2 = self.opponent.act(ob2)
        action2 = np.hstack([action, a2])
        return super().step(action2)


from gymnasium.envs.registration import register

try:
    register(
        id="Hockey-v0",
        entry_point="laserhockey.hockey_env:HockeyEnv",
        kwargs={"mode": 0},
    )
    register(
        id="Hockey-One-v0",
        entry_point="laserhockey.hockey_env:HockeyEnv_BasicOpponent",
        kwargs={"mode": 0, "weak_opponent": False},
    )
except Exception as e:
    print(e)
