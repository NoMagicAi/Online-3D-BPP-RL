from gym.envs.registration import register

# This tells the gym library how to create 'Bpp-v0'
register(
    id='Bpp-v0',
    entry_point='envs.bpp0:PackingGame', 
)