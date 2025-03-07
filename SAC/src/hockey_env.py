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
        reward: str = "basic",
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

        self.first_touch_done = False
        self.first_touch_done_agent2 = False

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

        valid_rewards = {"basic", "middle", "advanced", "01", "02", "03", "04", "05", "defensive", "aggressive", "counterattack", "possession"}

        if reward not in valid_rewards:
            raise ValueError(f"reward must be one of {valid_rewards}, got {reward}")
        self.reward = reward

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

    # ------------------ SYMMETRY HELPER METHODS ------------------
    # These methods implement a horizontal mirroring transformation.
    # Because the game is symmetric with respect to the horizontal axis,
    # a state observed above the center (with a given y component, angle, etc.)
    # has an equivalent mirror image below the center if we flip the sign
    # of the y–related components. Similarly, an action that moves the agent
    # upward (positive y) should be mirrored by an action that moves downward
    # (negative y) when the state is mirrored.
    #
    # Note: For a state vector (of 18 dims), we assume the following layout:
    #   Player 1: indices 0: x, 1: y, 2: angle, 3: x vel, 4: y vel, 5: ang vel
    #   Player 2: indices 6: x, 7: y, 8: angle, 9: x vel, 10: y vel, 11: ang vel
    #   Puck: indices 12: x, 13: y, 14: x vel, 15: y vel
    #   Keep mode flags: indices 16 and 17 (unchanged)
    #
    # For an action vector (8 dims: 4 for each agent):
    #   Agent: indices 0: x change, 1: y change, 2: angle change, 3: shoot (unchanged)
    #

    def mirror_state(self, state: np.ndarray) -> np.ndarray:
        mirrored = state.copy()

        # Player 1: flip y position, y velocity, angle, and angular velocity
        mirrored[1] = -mirrored[1]  # y position
        mirrored[4] = -mirrored[4]  # y velocity
        mirrored[2] = -mirrored[2]  # angle
        mirrored[5] = -mirrored[5]  # angular velocity

        # Player 2: indices 7, 10, 8, 11
        mirrored[7] = -mirrored[7]
        mirrored[10] = -mirrored[10]
        mirrored[8] = -mirrored[8]
        mirrored[11] = -mirrored[11]

        # Puck: flip y position and y velocity (indices 13 and 15)
        mirrored[13] = -mirrored[13]
        mirrored[15] = -mirrored[15]

        # Keep mode flags (indices 16, 17) remain unchanged
        return mirrored

    def mirror_action(self, action: np.ndarray) -> np.ndarray:
        mirrored = action.copy()
        if len(mirrored) == 4:
            # Single agent action: indices 0: x, 1: y, 2: angle, 3: shoot
            mirrored[1] = -mirrored[1]  # flip y
            mirrored[2] = -mirrored[2]  # flip angle
        elif len(mirrored) == 8:
            # Full action vector for both agents:
            # For agent 1 (indices 0-3): flip y (index 1) and angle (index 2)
            mirrored[1] = -mirrored[1]
            mirrored[2] = -mirrored[2]
            # For agent 2 (indices 4-7): flip y (index 5) and angle (index 6)
            mirrored[5] = -mirrored[5]
            mirrored[6] = -mirrored[6]
        else:
            raise ValueError(f"Unexpected action dimension: {len(mirrored)}")
        return mirrored

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

    def _get_reward_basic(self, info: Dict[str, Any]) -> float:
        """Exactly your existing 'basic' reward shaping or minimal shaping."""
        r = self._compute_reward()
        r += info["reward_closeness_to_puck"]  # or your existing proxy terms
        return float(r)

    def _get_reward_basic_two(self, info_two: Dict[str, Any]) -> float:
        """Basic reward for agent two (mirrored)."""
        # Typically you do -_compute_reward() for agent2, or your existing logic.
        r = -self._compute_reward()
        r += info_two["reward_closeness_to_puck"]
        return float(r)

    def _get_reward_01(self, info):
        return (
            info["reward_closeness_to_puck"] * 0.5
            + info["reward_touch_puck"] * 2.0
            + info["reward_puck_direction"] * 1.0
            + self._compute_reward()
        )

    def _get_reward_01_two(self, info_two):
        r = -self._compute_reward()
        r += info_two["reward_closeness_to_puck"] * 0.5
        r += info_two["reward_touch_puck"] * 2.0
        r += info_two["reward_puck_direction"] * 1.0
        return r

    def _get_reward_02(self, info: Dict[str, Any]) -> float:
        """
        A more nuanced reward for player one.

        Combines:
          - Win/loss signal (from _compute_reward)
          - Closeness to puck (penalty if far away)
          - Bonus for touching the puck
          - Bonus for the puck moving in the correct (attacking) direction
          - Bonus for making progress toward the opponent's goal

        Note: CENTER_X and W are defined globally.
        """
        base = self._compute_reward()
        closeness = info["reward_closeness_to_puck"]
        touch = info["reward_touch_puck"]
        direction = info["reward_puck_direction"]
        # For player one, the further right the puck is (i.e. its x > CENTER_X), the more progress.
        puck_progress = max(0, (self.puck.position[0] - CENTER_X) / (W / 2))
        # Combine with chosen weights:
        reward = (
            base + 0.3 * closeness + 3.0 * touch + 1.5 * direction + 2.0 * puck_progress
        )
        return float(reward)


    def _get_reward_defensive_two(self, info_two: Dict[str, Any]) -> float:
        """
        Mirror of defensive reward for player 2:
        1. Stay between puck and own goal (right side)
        2. Block shots
        3. Clear puck away from goal
        4. Don't over-commit backward
        """
        base = -self._compute_reward()  # Mirror win/loss
        
        # Get positions and velocities
        puck_pos = np.array([self.puck.position[0], self.puck.position[1]])
        player_pos = np.array([self.player2.position[0], self.player2.position[1]])
        goal_pos = np.array([W, CENTER_Y])  # Their goal is on right side
        
        # 1. Positioning reward: higher when player is between puck and goal
        if puck_pos[0] > CENTER_X:  # Puck in defensive half
            puck_to_goal = goal_pos - puck_pos
            player_to_goal = goal_pos - player_pos
            positioning_reward = np.dot(puck_to_goal, player_to_goal) / (np.linalg.norm(puck_to_goal) * np.linalg.norm(player_to_goal))
        else:
            positioning_reward = 0
        
        # 2. Shot blocking: reward for being close when puck moves toward goal
        if self.puck.linearVelocity[0] > 0:  # Puck moving toward own goal
            blocking_reward = -info_two["reward_closeness_to_puck"] * 2.0
        else:
            blocking_reward = 0
            
        # 3. Clearing reward: bonus for hitting puck away from goal
        clearing_reward = max(0, -self.puck.linearVelocity[0]) if info_two["reward_touch_puck"] > 0 else 0
        
        # 4. Position penalty: discourage going too far back
        position_penalty = -max(0, (CENTER_X - player_pos[0])) / (W/2)
        
        reward = (base 
                + 2.0 * positioning_reward 
                + blocking_reward 
                + clearing_reward 
                + 0.5 * position_penalty)
        
        return float(reward)

    def _get_reward_aggressive_two(self, info_two: Dict[str, Any]) -> float:
        """
        Mirror of aggressive reward for player 2:
        1. Forward positioning (toward left)
        2. Direct shots on goal (left goal)
        3. High shot velocity
        4. Quick possession recovery
        """
        base = -self._compute_reward()
        
        # 1. Forward positioning bonus (for player 2, forward is toward left)
        player_pos = np.array([self.player2.position[0], self.player2.position[1]])
        forward_bonus = max(0, (CENTER_X - player_pos[0])) / (W/2)
        
        # 2. Shot direction reward (toward left goal)
        puck_vel = np.array([self.puck.linearVelocity[0], self.puck.linearVelocity[1]])
        if np.linalg.norm(puck_vel) > 0:
            goal_dir = np.array([-1, 0])  # Vector pointing to opponent's goal (left)
            shot_alignment = np.dot(puck_vel, goal_dir) / np.linalg.norm(puck_vel)
            shot_reward = max(0, shot_alignment) if info_two["reward_touch_puck"] > 0 else 0
        else:
            shot_reward = 0
        
        # 3. Shot power reward
        shot_power = min(1.0, np.linalg.norm(puck_vel) / MAX_PUCK_SPEED)
        
        # 4. Quick possession reward
        possession_reward = 2.0 if info_two["reward_touch_puck"] > 0 else -0.1
        
        reward = (base 
                + 2.0 * forward_bonus
                + 3.0 * shot_reward
                + 2.0 * shot_power
                + possession_reward)
        
        return float(reward)

    def _get_reward_defensive(self, info: Dict[str, Any]) -> float:
        """
        Defensive reward that encourages:
        1. Staying between puck and own goal when puck is in defensive half
        2. Blocking shots
        3. Clearing the puck away from goal
        4. Not over-committing forward
        """
        base = self._compute_reward()  # Win/loss still matters
        
        # Get positions and velocities
        puck_pos = np.array([self.puck.position[0], self.puck.position[1]])
        player_pos = np.array([self.player1.position[0], self.player1.position[1]])
        goal_pos = np.array([0, CENTER_Y])  # Own goal position
        
        # 1. Positioning reward: higher when player is between puck and goal
        if puck_pos[0] < CENTER_X:  # Puck in defensive half
            puck_to_goal = goal_pos - puck_pos
            player_to_goal = goal_pos - player_pos
            positioning_reward = np.dot(puck_to_goal, player_to_goal) / (np.linalg.norm(puck_to_goal) * np.linalg.norm(player_to_goal))
        else:
            positioning_reward = 0
        
        # 2. Shot blocking: reward for being close to puck when it's moving toward goal
        if self.puck.linearVelocity[0] < 0:  # Puck moving toward own goal
            blocking_reward = -info["reward_closeness_to_puck"] * 2.0
        else:
            blocking_reward = 0
            
        # 3. Clearing reward: bonus for hitting puck away from goal
        clearing_reward = max(0, self.puck.linearVelocity[0]) if info["reward_touch_puck"] > 0 else 0
        
        # 4. Position penalty: discourage going too far forward
        position_penalty = -max(0, (player_pos[0] - CENTER_X)) / (W/2)
        
        reward = (base 
                + 2.0 * positioning_reward 
                + blocking_reward 
                + clearing_reward 
                + 0.5 * position_penalty)
        
        return float(reward)

    def _get_reward_aggressive(self, info: Dict[str, Any]) -> float:
        """
        Aggressive reward that encourages:
        1. Forward positioning
        2. Direct shots on goal
        3. High shot velocity
        4. Quick possession recovery
        """
        base = self._compute_reward()
        
        # 1. Forward positioning bonus
        player_pos = np.array([self.player1.position[0], self.player1.position[1]])
        forward_bonus = max(0, (player_pos[0] - CENTER_X)) / (W/2)
        
        # 2. Shot direction reward
        puck_vel = np.array([self.puck.linearVelocity[0], self.puck.linearVelocity[1]])
        if np.linalg.norm(puck_vel) > 0:
            goal_dir = np.array([1, 0])  # Vector pointing to opponent's goal
            shot_alignment = np.dot(puck_vel, goal_dir) / np.linalg.norm(puck_vel)
            shot_reward = max(0, shot_alignment) if info["reward_touch_puck"] > 0 else 0
        else:
            shot_reward = 0
        
        # 3. Shot power reward
        shot_power = min(1.0, np.linalg.norm(puck_vel) / MAX_PUCK_SPEED)
        
        # 4. Quick possession reward
        possession_reward = 2.0 if info["reward_touch_puck"] > 0 else -0.1
        
        reward = (base 
                + 2.0 * forward_bonus
                + 3.0 * shot_reward
                + 2.0 * shot_power
                + possession_reward)
        
        return float(reward)

    def _get_reward_possession(self, info: Dict[str, Any]) -> float:
        """
        Possession-based reward that encourages:
        1. Maintaining control of the puck
        2. Safe passes/movements
        3. Position control
        4. Puck protection
        """
        base = self._compute_reward()
        
        # 1. Possession time reward
        possession_time = self.player1_has_puck / MAX_TIME_KEEP_PUCK
        
        # 2. Safe movement reward: higher reward for keeping puck away from opponent
        puck_pos = np.array([self.puck.position[0], self.puck.position[1]])
        opp_pos = np.array([self.player2.position[0], self.player2.position[1]])
        safety_distance = np.linalg.norm(puck_pos - opp_pos)
        safety_reward = min(1.0, safety_distance / (W/2)) if self.player1_has_puck > 0 else 0
        
        # 3. Position control: reward for keeping puck in advantageous positions
        position_control = max(0, (puck_pos[0] - CENTER_X) / (W/2))
        
        # 4. Puck protection: penalize high-risk movements
        puck_vel = np.array([self.puck.linearVelocity[0], self.puck.linearVelocity[1]])
        risk_penalty = -min(1.0, np.linalg.norm(puck_vel) / MAX_PUCK_SPEED) if self.player1_has_puck > 0 else 0
        
        reward = (base
                + 3.0 * possession_time
                + 2.0 * safety_reward
                + position_control
                + 0.5 * risk_penalty)
        
        return float(reward)

    def _get_reward_counterattack(self, info: Dict[str, Any]) -> float:
        """
        Counter-attacking reward that encourages:
        1. Quick transitions after gaining possession
        2. Fast breaks toward goal
        3. Efficient puck movement
        4. Strategic positioning for interceptions
        """
        base = self._compute_reward()
        
        # 1. Transition reward: bonus for quickly moving puck forward after gaining possession
        transition_reward = 0
        if info["reward_touch_puck"] > 0 and self.puck.linearVelocity[0] > 0:
            transition_reward = min(1.0, self.puck.linearVelocity[0] / MAX_PUCK_SPEED)
        
        # 2. Fast break reward
        puck_vel = np.array([self.puck.linearVelocity[0], self.puck.linearVelocity[1]])
        speed_reward = min(1.0, np.linalg.norm(puck_vel) / MAX_PUCK_SPEED)
        
        # 3. Movement efficiency: reward direct paths to goal when in possession
        if self.player1_has_puck > 0:
            player_pos = np.array([self.player1.position[0], self.player1.position[1]])
            goal_pos = np.array([W, CENTER_Y])
            path_directness = 1.0 - abs(player_pos[1] - goal_pos[1]) / (H/2)
            efficiency_reward = path_directness
        else:
            efficiency_reward = 0
        
        # 4. Interception positioning: reward being in good position to intercept passes
        player_pos = np.array([self.player1.position[0], self.player1.position[1]])
        if self.puck.linearVelocity[0] < 0:  # Puck moving toward our side
            intercept_pos = player_pos[0] - self.puck.position[0]
            positioning_reward = 1.0 if 0 < intercept_pos < W/4 else 0
        else:
            positioning_reward = 0
        
        reward = (base
                + 3.0 * transition_reward
                + 2.0 * speed_reward
                + efficiency_reward
                + 2.0 * positioning_reward)
        
        return float(reward)
    def _get_reward_possession_two(self, info_two: Dict[str, Any]) -> float:
        """
        Mirror of possession reward for player 2:
        1. Maintain control of puck
        2. Safe passes/movements
        3. Position control
        4. Puck protection
        """
        base = -self._compute_reward()
        
        # 1. Possession time reward
        possession_time = self.player2_has_puck / MAX_TIME_KEEP_PUCK
        
        # 2. Safe movement reward: higher for keeping puck away from opponent
        puck_pos = np.array([self.puck.position[0], self.puck.position[1]])
        opp_pos = np.array([self.player1.position[0], self.player1.position[1]])
        safety_distance = np.linalg.norm(puck_pos - opp_pos)
        safety_reward = min(1.0, safety_distance / (W/2)) if self.player2_has_puck > 0 else 0
        
        # 3. Position control: reward for keeping puck in advantageous positions
        position_control = max(0, (CENTER_X - puck_pos[0]) / (W/2))
        
        # 4. Puck protection: penalize high-risk movements
        puck_vel = np.array([self.puck.linearVelocity[0], self.puck.linearVelocity[1]])
        risk_penalty = -min(1.0, np.linalg.norm(puck_vel) / MAX_PUCK_SPEED) if self.player2_has_puck > 0 else 0
        
        reward = (base
                + 3.0 * possession_time
                + 2.0 * safety_reward
                + position_control
                + 0.5 * risk_penalty)
        
        return float(reward)

    def _get_reward_counterattack_two(self, info_two: Dict[str, Any]) -> float:
        """
        Mirror of counter-attacking reward for player 2:
        1. Quick transitions after gaining possession
        2. Fast breaks toward goal (left)
        3. Efficient puck movement
        4. Strategic positioning for interceptions
        """
        base = -self._compute_reward()
        
        # 1. Transition reward: bonus for quickly moving puck forward (left) after gaining possession
        transition_reward = 0
        if info_two["reward_touch_puck"] > 0 and self.puck.linearVelocity[0] < 0:
            transition_reward = min(1.0, -self.puck.linearVelocity[0] / MAX_PUCK_SPEED)
        
        # 2. Fast break reward
        puck_vel = np.array([self.puck.linearVelocity[0], self.puck.linearVelocity[1]])
        speed_reward = min(1.0, np.linalg.norm(puck_vel) / MAX_PUCK_SPEED)
        
        # 3. Movement efficiency: reward direct paths to goal when in possession
        if self.player2_has_puck > 0:
            player_pos = np.array([self.player2.position[0], self.player2.position[1]])
            goal_pos = np.array([0, CENTER_Y])  # Left goal
            path_directness = 1.0 - abs(player_pos[1] - goal_pos[1]) / (H/2)
            efficiency_reward = path_directness
        else:
            efficiency_reward = 0
        
        # 4. Interception positioning: reward being in good position to intercept passes
        player_pos = np.array([self.player2.position[0], self.player2.position[1]])
        if self.puck.linearVelocity[0] > 0:  # Puck moving toward our side
            intercept_pos = self.puck.position[0] - player_pos[0]
            positioning_reward = 1.0 if 0 < intercept_pos < W/4 else 0
        else:
            positioning_reward = 0
        
        reward = (base
                + 3.0 * transition_reward
                + 2.0 * speed_reward
                + efficiency_reward
                + 2.0 * positioning_reward)
        
        return float(reward)

    def _get_reward_02_two(self, info_two: Dict[str, Any]) -> float:
        """
        A symmetric reward for player two.

        It mirrors the logic for player one:
          - Uses the mirrored win/loss signal (-_compute_reward())
          - Uses the mirrored closeness, touch, and direction rewards from info_two
          - Rewards progress if the puck is moving toward player two's goal
            (i.e. its x < CENTER_X)
        """
        base = -self._compute_reward()  # mirror the win/loss signal
        closeness = info_two["reward_closeness_to_puck"]
        touch = info_two["reward_touch_puck"]
        direction = info_two["reward_puck_direction"]
        # For player two, we reward when the puck has moved left of center.
        puck_progress = max(0, (CENTER_X - self.puck.position[0]) / (W / 2))
        reward = (
            base + 0.3 * closeness + 3.0 * touch + 1.5 * direction + 2.0 * puck_progress
        )
        return float(reward)

    def _get_reward_04(self, info: Dict[str, Any]) -> float:
        """
        Enhanced reward for player one with:
        1. Progressive positioning bonus
        2. Sustained possession bonus
        3. Angle-aligned shot quality
        4. Dynamic defensive positioning
        5. Early touch incentive
        """
        base = self._compute_reward()

        # 1. Progressive positioning - normalized puck progress toward opponent goal
        puck_progress = max(
            0.0, (self.puck.position[0] - CENTER_X) / (W / 2 - CENTER_X)
        )

        # 2. Sustained possession - bonus per step while controlling puck
        sustained_possession = 0.1 * (self.player1_has_puck / MAX_TIME_KEEP_PUCK)

        # 3. Angle-aligned shot quality - combines speed and accuracy
        shot_speed = self.puck.linearVelocity[0] / MAX_PUCK_SPEED  # Normalized
        y_alignment = 1.0 - abs(self.puck.position[1] - CENTER_Y) / (H / 2)  # 1=center
        shot_quality = shot_speed * y_alignment

        # 4. Dynamic defensive positioning - reduce penalty if puck is moving away
        defensive_penalty = info["reward_closeness_to_puck"]
        if self.puck.linearVelocity[0] > 0:  # Puck moving toward opponent side
            defensive_penalty *= 0.2  # Reduce penalty by 80%

        # 5. Early touch bonus - incentivize quick puck acquisition
        early_touch = (
            info["reward_touch_puck"]
            * (self.max_timesteps - self.time)
            / self.max_timesteps
        )

        reward = (
            base
            + 2.0 * puck_progress
            + sustained_possession
            + 1.5 * shot_quality
            + defensive_penalty
            + 0.5 * early_touch
            - 0.01  # Small constant penalty to encourage urgency
        )
        return float(reward)

    def _get_reward_04_two(self, info_two: Dict[str, Any]) -> float:
        """Mirrored version for player two"""
        base = -self._compute_reward()

        puck_progress = max(
            0.0, (CENTER_X - self.puck.position[0]) / (W / 2 - CENTER_X)
        )
        sustained_possession = 0.1 * (self.player2_has_puck / MAX_TIME_KEEP_PUCK)
        shot_speed = -self.puck.linearVelocity[0] / MAX_PUCK_SPEED
        y_alignment = 1.0 - abs(self.puck.position[1] - CENTER_Y) / (H / 2)
        shot_quality = shot_speed * y_alignment
        defensive_penalty = info_two["reward_closeness_to_puck"]
        if self.puck.linearVelocity[0] < 0:  # Puck moving toward player two's side
            defensive_penalty *= 0.2
        early_touch = (
            info_two["reward_touch_puck"]
            * (self.max_timesteps - self.time)
            / self.max_timesteps
        )

        reward = (
            base
            + 2.0 * puck_progress
            + sustained_possession
            + 1.5 * shot_quality
            + defensive_penalty
            + 0.5 * early_touch
            - 0.01
        )
        return float(reward)

    def _get_reward_05(self, info: Dict[str, Any]) -> float:
        """
        Enhanced reward for player one with:
        - Base win/loss reward.
        - A bonus proportional to the puck closeness (scaled by 5).
        - A small penalty if the puck is not touched.
        - A one-time bonus at the first touch scaled by (max_timesteps - current_step).
        """
        base = self._compute_reward()
        closeness = info["reward_closeness_to_puck"]
        touched = info[
            "reward_touch_puck"
        ]  # Expected to be 1.0 if the puck is touched; 0 otherwise.

        bonus = 0.0
        # If the puck is touched for the first time during this episode, add a bonus.
        if (not self.first_touch_done) and (touched == 1.0):
            bonus = 0.1 * (self.max_timesteps - self.time)
            self.first_touch_done = True

        reward = base + 5.0 * closeness - (1 - touched) * 0.1 + bonus
        return float(reward)

    def _get_reward_05_two(self, info_two: Dict[str, Any]) -> float:
        """
        Mirrored enhanced reward for player two.
        """
        base = -self._compute_reward()  # Mirror the win/loss signal.
        closeness = info_two["reward_closeness_to_puck"]
        touched = info_two["reward_touch_puck"]

        bonus = 0.0
        if (not self.first_touch_done_agent2) and (touched == 1.0):
            bonus = 0.1 * (self.max_timesteps - self.time)
            self.first_touch_done_agent2 = True

        reward = base + 5.0 * closeness - (1 - touched) * 0.1 + bonus
        return float(reward)

    def _get_reward_middle(self, info: Dict[str, Any]) -> float:
        pass  # implement this!

    def _get_reward_middle_two(self, info_two: Dict[str, Any]) -> float:
        pass  # implement this!

    # --------------------------------------------------------------
    # 3) ADVANCED REWARD
    #
    # --------------------------------------------------------------

    def _get_reward_03(self, info: Dict[str, Any]) -> float:
        """
        A more nuanced reward for player one that augments the basic reward as follows:

         A) env_reward: the base reward (e.g. win/loss)
         B) A bonus proportional to how close the agent is to the puck.
         C) A small per-step penalty if the agent does not touch the puck.
         D) A bonus if the agent touches the puck for the first time, scaled by how early the touch occurs.
         E) A bonus for the puck moving in the desired direction.

        The timing bonus is computed as (max_timesteps - current_step) so that an early touch yields
        a higher bonus.
        """
        base = self._compute_reward()  # (A)
        closeness = info["reward_closeness_to_puck"]  # (B)
        touched = info["reward_touch_puck"]  # 1.0 if touched; 0 otherwise
        direction = info[
            "reward_puck_direction"
        ]  # bonus if moving in the right direction

        # (D): Bonus if the puck is touched for the first time.
        # (max_timesteps - self.time) is larger if the touch occurs early.
        timing_bonus = touched * (self.max_timesteps - self.time)

        # Combine with chosen weights:
        reward = (
            base
            + 4.0 * closeness
            - 0.02 * (1 - touched)
            + 0.15 * timing_bonus
            + 1.0 * direction
        )
        return float(reward)

    def _get_reward_03_two(self, info_two: Dict[str, Any]) -> float:
        """
        A symmetric reward for player two.

        The idea is the same as for player one except that:
          - The base reward is mirrored (i.e. negative of _compute_reward())
          - The directional bonus should be appropriate for player two.
        """
        base = -self._compute_reward()  # mirror win/loss signal
        closeness = info_two["reward_closeness_to_puck"]
        touched = info_two["reward_touch_puck"]
        direction = info_two["reward_puck_direction"]

        timing_bonus = touched * (self.max_timesteps - self.time)

        reward = (
            base
            + 4.0 * closeness
            - 0.02 * (1 - touched)
            + 0.15 * timing_bonus
            + 1.0 * direction
        )
        return float(reward)

    def _get_reward_advanced(self, info: Dict[str, Any]) -> float:
        # Base win/loss reward.
        base = self._compute_reward()

        # Proxy rewards.
        closeness = info["reward_closeness_to_puck"]
        touch = info["reward_touch_puck"]
        direction = info["reward_puck_direction"]

        # Progress toward the opponent's goal:
        puck_progress = max(0, (self.puck.position[0] - CENTER_X) / (W / 2))

        # Sustained possession bonus.
        sustained_possession = 0.1 * (self.player1_has_puck / MAX_TIME_KEEP_PUCK)

        # Shot quality computation:
        vx, vy = self.puck.linearVelocity[0], self.puck.linearVelocity[1]
        shot_speed = (
            np.sqrt(vx * vx + vy * vy) / MAX_PUCK_SPEED
        )  # normalized shot speed
        theta_v = math.atan2(vy, vx)  # direction of the puck's velocity

        # Compute the direction toward the goal.
        goal_center = (W, CENTER_Y)  # assuming the opponent's goal is at (W, CENTER_Y)
        dx = goal_center[0] - self.puck.position[0]
        dy = goal_center[1] - self.puck.position[1]
        theta_goal = math.atan2(dy, dx)

        # Angular error between shot direction and goal direction.
        angle_error = abs(theta_v - theta_goal)

        # Adjust penalty for angular error based on proximity to the top or bottom wall.
        # If the puck is near a wall, reduce the impact of the angle error (bank shot).
        distance_to_wall = min(self.puck.position[1], H - self.puck.position[1])
        threshold = H / 4  # threshold distance under which bank shots are likely
        bank_factor = (
            distance_to_wall / threshold if distance_to_wall < threshold else 1.0
        )

        # Shot quality: higher when shot speed is high and the (weighted) angle error is low.
        shot_quality = shot_speed * (1 - bank_factor * angle_error / math.pi)

        # Early touch bonus: incentivizes quick puck acquisition.
        early_touch = touch * (self.max_timesteps - self.time) / self.max_timesteps

        # Combine all components with chosen weights.
        reward = (
            base
            + 0.3 * closeness
            + 3.0 * touch
            + 1.5 * direction
            + 2.0 * puck_progress
            + sustained_possession
            + 1.5 * shot_quality
            + 0.5 * early_touch
            - 0.01  # small constant penalty for urgency
        )

        return float(reward)

    def _get_reward_advanced_two(self, info_two: Dict[str, Any]) -> float:
        # Mirror the base win/loss reward.
        base = -self._compute_reward()

        # Proxy rewards from info for player two.
        closeness = info_two["reward_closeness_to_puck"]
        touch = info_two["reward_touch_puck"]
        direction = info_two["reward_puck_direction"]

        # For player two, progress is measured as how far the puck has moved toward the left side.
        puck_progress = max(0, (CENTER_X - self.puck.position[0]) / (W / 2))

        # Sustained possession bonus for player two.
        sustained_possession = 0.1 * (self.player2_has_puck / MAX_TIME_KEEP_PUCK)

        # Shot quality computation for player two:
        vx, vy = self.puck.linearVelocity[0], self.puck.linearVelocity[1]
        # Compute normalized shot speed (same for both players).
        shot_speed = np.sqrt(vx * vx + vy * vy) / MAX_PUCK_SPEED
        theta_v = math.atan2(vy, vx)  # Direction of puck's velocity.

        # For player two, assume its goal is at the left side, i.e. at (0, CENTER_Y).
        goal_center = (0, CENTER_Y)
        dx = goal_center[0] - self.puck.position[0]
        dy = goal_center[1] - self.puck.position[1]
        theta_goal = math.atan2(dy, dx)

        # Angular error between the puck's velocity and the direction to player two's goal.
        angle_error = abs(theta_v - theta_goal)

        # Adjust the angular error based on proximity to the top or bottom wall
        # (to reduce penalty when bank shots are likely).
        distance_to_wall = min(self.puck.position[1], H - self.puck.position[1])
        threshold = H / 4  # Threshold distance below which bank shots are considered.
        bank_factor = (
            distance_to_wall / threshold if distance_to_wall < threshold else 1.0
        )

        # Shot quality: higher when the puck moves fast and its (weighted) angular error is small.
        shot_quality = shot_speed * (1 - bank_factor * angle_error / math.pi)

        # Early touch bonus: incentivizes acquiring the puck quickly.
        early_touch = touch * (self.max_timesteps - self.time) / self.max_timesteps

        reward = (
            base
            + 0.3 * closeness
            + 3.0 * touch
            + 1.5 * direction
            + 2.0 * puck_progress
            + sustained_possession
            + 1.5 * shot_quality
            + 0.5 * early_touch
            - 0.01  # Small constant penalty to encourage urgency.
        )

        return float(reward)

    def get_reward(self, info: Dict[str, Any]) -> float:
        """Return the reward for player one (the 'main' agent)."""
        if self.reward == "basic":
            return self._get_reward_basic(info)
        elif self.reward == "middle":
            return self._get_reward_middle(info)
        elif self.reward == "advanced":
            return self._get_reward_advanced(info)
        elif self.reward == "01":
            return self._get_reward_01(info)
        elif self.reward == "02":
            return self._get_reward_02(info)
        elif self.reward == "03":
            return self._get_reward_03(info)
        elif self.reward == "04":
            return self._get_reward_04(info)
        elif self.reward == "05":
            return self._get_reward_05(info)
        elif self.reward == "defensive":
            return self._get_reward_defensive(info)
        elif self.reward == "aggressive":
            return self._get_reward_aggressive(info)
        elif self.reward == "possession":
            return self._get_reward_possession(info)
        elif self.reward == "counterattack":
            return self._get_reward_counterattack(info)
        else:
            # Should not happen if we validated reward in constructor
            return 0.0

    def get_reward_agent_two(self, info_two: Dict[str, Any]) -> float:
        """Return the reward for player two (mirrored)."""
        if self.reward == "basic":
            return self._get_reward_basic_two(info_two)
        elif self.reward == "middle":
            return self._get_reward_middle_two(info_two)
        elif self.reward == "advanced":
            return self._get_reward_advanced_two(info_two)
        elif self.reward == "01":
            return self._get_reward_01_two(info_two)
        elif self.reward == "02":
            return self._get_reward_02_two(info_two)
        elif self.reward == "03":
            return self._get_reward_03_two(info_two)
        elif self.reward == "04":
            return self._get_reward_04_two(info_two)
        elif self.reward == "05":
            return self._get_reward_05_two(info_two)
        elif self.reward == "defensive":
            return self._get_reward_defensive(info_two)
        elif self.reward == "aggressive":
            return self._get_reward_aggressive(info_two)
        elif self.reward == "possession":
            return self._get_reward_possession_two(info_two)
        elif self.reward == "counterattack":
            return self._get_reward_counterattack_two(info_two)
        else:
            return 0.0

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
    def __init__(self, env, player=1, reward="basic"):
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


class BasicDefenseOpponent:
    """
    In attack mode, the opponent uses a defensive strategy:
    It always runs straight toward its own net.
    In the mirrored observation space, this fixed target is set to (-210/SCALE, 0).
    """

    def __init__(self, keep_mode=True):
        self.keep_mode = keep_mode
        self.kp = 0.5
        self.kd = 0.5

    def act(self, obs, verbose=False):
        # In the mirrored observation, the opponent's own state is given as:
        # [x position, y position, angle, x velocity, y velocity, angular velocity, ...]
        # We extract the first three values as position and angle.
        p = np.asarray([obs[0], obs[1], obs[2]])
        v = np.asarray(obs[3:6])
        # Set a fixed target: in mirrored coordinates, the opponent's net is at (-210/SCALE, 0)
        target_pos = np.array([-210.0 / SCALE, 0.0])
        target_angle = 0.0
        target = np.array([target_pos[0], target_pos[1], target_angle])
        error = target - p
        time_to_break = 0.1
        need_break = np.abs(error / (v + 0.01)) < np.array(
            [time_to_break, time_to_break, time_to_break * 10]
        )
        action = np.clip(
            error * np.array([self.kp, self.kp / 5, self.kp / 2])
            - v * need_break * np.array([self.kd, self.kd, self.kd]),
            -1,
            1,
        )
        shoot = 0.0  # Defense opponent does not shoot.
        if self.keep_mode:
            return np.hstack([action, [shoot]])
        else:
            return action


class BasicAttackOpponent:
    """
    In defense mode, the opponent uses an attacking strategy:
    It always runs toward the middle of the field and tries to shoot.
    In the mirrored observation space, the middle is at (0,0).
    """

    def __init__(self, keep_mode=True):
        self.keep_mode = keep_mode
        self.kp = 0.5
        self.kd = 0.5

    def act(self, obs, verbose=False):
        p = np.asarray([obs[0], obs[1], obs[2]])
        v = np.asarray(obs[3:6])
        # Target the middle of the field (0, 0) with zero target angle.
        target_pos = np.array([0.0, 0.0])
        target_angle = 0.0
        target = np.array([target_pos[0], target_pos[1], target_angle])
        error = target - p
        time_to_break = 0.1
        need_break = np.abs(error / (v + 0.01)) < np.array(
            [time_to_break, time_to_break, time_to_break * 10]
        )
        action = np.clip(
            error * np.array([self.kp, self.kp / 5, self.kp / 2])
            - v * need_break * np.array([self.kd, self.kd, self.kd]),
            -1,
            1,
        )
        # Attack opponent tries to shoot
        shoot = 1.0 if self.keep_mode else 0.0
        if self.keep_mode:
            return np.hstack([action, [shoot]])
        else:
            return action


class HockeyReplayEnv(HockeyEnv):
    """
    Extension of HockeyEnv that supports replaying recorded games.
    """

    def __init__(self, game_data=None):
        super().__init__()
        self.game_data = game_data
        self.current_round = 0
        self.current_step = 0
        self.replay_mode = False

    def load_game(self, game_data):
        """Load a recorded game for replay.

        Args:
            game_data (dict): Dictionary containing recorded game data with actions and observations
                             for each round.
        """
        self.game_data = game_data
        self.current_round = 0
        self.current_step = 0
        self.replay_mode = True

        # Reset to initial state of first round
        if f"observations_round_0" in game_data:
            initial_state = game_data["observations_round_0"][0]
            self.reset()
            self.set_state(initial_state)

    def replay_step(self):
        """Execute next step in the replay sequence.

        Returns:
            tuple: (observation, reward, done, truncated, info) for the replayed step,
                  or (None, None, True, False, {}) if replay is complete
        """
        if not self.replay_mode or self.game_data is None:
            raise RuntimeError("Not in replay mode or no game data loaded")

        # Check if we've reached the end of all rounds
        if self.current_round >= self.game_data["num_rounds"]:
            return None, None, True, False, {}

        # Get current round's actions and observations
        actions_key = f"actions_round_{self.current_round}"
        observations_key = f"observations_round_{self.current_round}"

        actions = self.game_data[actions_key]
        observations = self.game_data[observations_key]

        # Check if we've reached the end of current round
        if self.current_step >= len(actions):
            self.current_round += 1
            self.current_step = 0
            if self.current_round < self.game_data["num_rounds"]:
                # Set initial state of next round
                next_obs_key = f"observations_round_{self.current_round}"
                initial_state = self.game_data[next_obs_key][0]
                self.reset()
                self.set_state(initial_state)
            return self.replay_step()

        # Execute the recorded action
        action = actions[self.current_step]
        self.current_step += 1

        return self.step(action)

    def render_replay(self, mode="human", fps=50):
        """Render the entire replay from start to finish.

        Args:
            mode (str): Rendering mode ('human' or 'rgb_array')
            fps (int): Frames per second for replay
        """
        import time

        if not self.replay_mode or self.game_data is None:
            raise RuntimeError("Not in replay mode or no game data loaded")

        # Reset to beginning
        self.current_round = 0
        self.current_step = 0

        # Load initial state
        initial_state = self.game_data["observations_round_0"][0]
        self.reset()
        self.set_state(initial_state)

        frame_delay = 1.0 / fps
        done = False

        while not done:
            start_time = time.time()

            obs, reward, done, truncated, info = self.replay_step()
            if done:
                break

            self.render(mode=mode)

            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)


def replay_game(game_data, render_mode="human", fps=50):
    """Convenience function to replay a recorded game.

    Args:
        game_data (dict): Dictionary containing recorded game data
        render_mode (str): Rendering mode ('human' or 'rgb_array')
        fps (int): Frames per second for replay
    """
    env = HockeyReplayEnv()
    env.load_game(game_data)
    env.render_replay(mode=render_mode, fps=fps)
    env.close()


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
        entry_point="src.hockey_env:HockeyEnv",  # <-- changed!
        kwargs={"mode": 0},
    )
    register(
        id="Hockey-One-v0",
        entry_point="src.hockey_env:HockeyEnv_BasicOpponent",  # <-- changed!
        kwargs={"mode": 0, "weak_opponent": False},
    )
    register(
        id="Hockey-Better-v0",
        entry_point="src.hockey_env:HockeyEnv_BetterOpponent",
        kwargs={"mode": 0, "advanced": False},
    )
    register(
        id="Hockey-Advanced-v0",
        entry_point="src.hockey_env:HockeyEnv_BetterOpponent",
        kwargs={"mode": 0, "advanced": True},
    )
except Exception as e:
    print(e)
