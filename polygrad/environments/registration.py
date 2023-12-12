import gym

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperFullObs-v2',
        'entry_point': ('polygrad.environments.hopper:HopperFullObsEnv'),
    },
    {
        'id': 'HalfCheetahFullObs-v2',
        'entry_point': ('polygrad.environments.half_cheetah:HalfCheetahFullObsEnv'),
    },
    {
        'id': 'Walker2dFullObs-v2',
        'entry_point': ('polygrad.environments.walker2d:Walker2dFullObsEnv'),
    },
    {
        'id': 'AntFullObs-v2',
        'entry_point': ('polygrad.environments.ant:AntFullObsEnv'),
    },

    {
        'id': 'ToyEnv-v0',
        'entry_point': ('polygrad.environments.toyenv.toyenv:ToyEnv'),
        'max_episode_steps': 1000,
    },
    {
        'id': 'SimpleMaze-v0',
        'entry_point': ('polygrad.environments.simple_maze.simple_maze:SimpleMaze'),
        'max_episode_steps': 1000,
    },
    {
        'id': 'CustomPendulum-v0',
        'entry_point': ('polygrad.environments.classic_control.pendulum:PendulumEnv'),
        'max_episode_steps': 1000,
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ polygrad/environments/registration ] WARNING: not registering polygrad environments')
        return tuple()