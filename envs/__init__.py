from gym.envs.registration import register

register(
    id="NBottleneckClass-v0",
    entry_point="envs.NBottleneckClass:NBottleneckClass",
)

register(
    id="NCycleClass-v0",
    entry_point="envs.NCycleClass:NCycleClass",
)

register(
        id="ScaleClass-v0",
    entry_point="envs.ScaleClass:ScaleClass",
)
