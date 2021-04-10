ALE = 2
ALE_TIMELIMIT = 27000
SPATIAL = 1

# ENV_INFO
# -- abbreviation
# -- env type (unused)
# -- obs type (unused)
# -- states are countable (unused)
# -- difficulty_ramping (unused)
# -- level (unused)
# -- initial_difficulty (unused)
# -- maximum timesteps per episode

ENV_INFO = {
    'ALE-Adventure': ('ALE01V0', ALE, SPATIAL, False, 'AdventureNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-AirRaid': ('ALE02V0', ALE, SPATIAL, False, 'AirRaidNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Alien': ('ALE03V0', ALE, SPATIAL, False, 'AlienNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Amidar': ('ALE04V0', ALE, SPATIAL, False, 'AmidarNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Assault': ('ALE05V0', ALE, SPATIAL, False, 'AssaultNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Asterix': ('ALE06V0', ALE, SPATIAL, False, 'AsterixNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Asteroids': ('ALE07V0', ALE, SPATIAL, False, 'AsteroidsNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Atlantis': ('ALE08V0', ALE, SPATIAL, False, 'AtlantisNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-BankHeist': ('ALE09V0', ALE, SPATIAL, False, 'BankHeistNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-BattleZone': ('ALE10V0', ALE, SPATIAL, False, 'BattleZoneNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-BeamRider': ('ALE11V0', ALE, SPATIAL, False, 'BeamRiderNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Berzerk': ('ALE12V0', ALE, SPATIAL, False, 'BerzerkNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Bowling': ('ALE13V0', ALE, SPATIAL, False, 'BowlingNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Boxing': ('ALE14V0', ALE, SPATIAL, False, 'BoxingNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Breakout': ('ALE15V0', ALE, SPATIAL, False, 'BreakoutNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Carnival': ('ALE16V0', ALE, SPATIAL, False, 'CarnivalNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Centipede': ('ALE17V0', ALE, SPATIAL, False, 'CentipedeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-ChopperCommand': ('ALE18V0', ALE, SPATIAL, False, 'ChopperCommandNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-CrazyClimber': ('ALE19V0', ALE, SPATIAL, False, 'CrazyClimberNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Defender': ('ALE20V0', ALE, SPATIAL, False, 'DefenderNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-DemonAttack': ('ALE21V0', ALE, SPATIAL, False, 'DemonAttackNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-DoubleDunk': ('ALE22V0', ALE, SPATIAL, False, 'DoubleDunkNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-ElevatorAction': ('ALE23V0', ALE, SPATIAL, False, 'ElevatorActionNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Enduro': ('ALE24V0', ALE, SPATIAL, False, 'EnduroNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-FishingDerby': ('ALE25V0', ALE, SPATIAL, False, 'FishingDerbyNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Freeway': ('ALE26V0', ALE, SPATIAL, False, 'FreewayNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Frostbite': ('ALE27V0', ALE, SPATIAL, False, 'FrostbiteNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Gopher': ('ALE28V0', ALE, SPATIAL, False, 'GopherNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Gravitar': ('ALE29V0', ALE, SPATIAL, False, 'GravitarNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Hero': ('ALE30V0', ALE, SPATIAL, False, 'HeroNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-IceHockey': ('ALE31V0', ALE, SPATIAL, False, 'IceHockeyNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Jamesbond': ('ALE32V0', ALE, SPATIAL, False, 'JamesbondNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-JourneyEscape': ('ALE33V0', ALE, SPATIAL, False, 'JourneyEscapeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Kaboom': ('ALE34V0', ALE, SPATIAL, False, 'KaboomNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Kangaroo': ('ALE35V0', ALE, SPATIAL, False, 'KangarooNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Krull': ('ALE36V0', ALE, SPATIAL, False, 'KrullNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-KungFuMaster': ('ALE37V0', ALE, SPATIAL, False, 'KungFuMasterNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-MontezumaRevenge': ('ALE38V0', ALE, SPATIAL, False, 'MontezumaRevengeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-MsPacman': ('ALE39V0', ALE, SPATIAL, False, 'MsPacmanNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-NameThisGame': ('ALE40V0', ALE, SPATIAL, False, 'NameThisGameNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Phoenix': ('ALE41V0', ALE, SPATIAL, False, 'PhoenixNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Pitfall': ('ALE42V0', ALE, SPATIAL, False, 'PitfallNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Pong': ('ALE43V0', ALE, SPATIAL, False, 'PongNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Pooyan': ('ALE44V0', ALE, SPATIAL, False, 'PooyanNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-PrivateEye': ('ALE45V0', ALE, SPATIAL, False, 'PrivateEyeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Qbert': ('ALE46V0', ALE, SPATIAL, False, 'QbertNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Riverraid': ('ALE47V0', ALE, SPATIAL, False, 'RiverraidNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-RoadRunner': ('ALE48V0', ALE, SPATIAL, False, 'RoadRunnerNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Robotank': ('ALE49V0', ALE, SPATIAL, False, 'RobotankNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Seaquest': ('ALE50V0', ALE, SPATIAL, False, 'SeaquestNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Skiing': ('ALE51V0', ALE, SPATIAL, False, 'SkiingNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Solaris': ('ALE52V0', ALE, SPATIAL, False, 'SolarisNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-SpaceInvaders': ('ALE53V0', ALE, SPATIAL, False, 'SpaceInvadersNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-StarGunner': ('ALE54V0', ALE, SPATIAL, False, 'StarGunnerNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Tennis': ('ALE55V0', ALE, SPATIAL, False, 'TennisNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-TimePilot': ('ALE56V0', ALE, SPATIAL, False, 'TimePilotNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Tutankham': ('ALE57V0', ALE, SPATIAL, False, 'TutankhamNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-UpNDown': ('ALE58V0', ALE, SPATIAL, False, 'UpNDownNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Venture': ('ALE59V0', ALE, SPATIAL, False, 'VentureNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-VideoPinball': ('ALE60V0', ALE, SPATIAL, False, 'VideoPinballNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-WizardOfWor': ('ALE61V0', ALE, SPATIAL, False, 'WizardOfWorNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-YarsRevenge': ('ALE62V0', ALE, SPATIAL, False, 'YarsRevengeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Zaxxon': ('ALE63V0', ALE, SPATIAL, False, 'ZaxxonNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
}

# Teacher Models (to load previously saved models as teachers):
# <Game Name>: (<Model directory>, <Model subdirectory (seed)>, <Checkpoint (timesteps)>)

# Example: ALE24V0_EG_000_20201105-130625/0/model-6000000.ckpt will be loaded from "checkpoints" folder when environment
# is Enduro

TEACHER = {
    'ALE-Enduro': ('ALE24V0_EG_000_20201105-130625', '0', 6000e3),
    'ALE-Freeway': ('ALE26V0_EG_000_20201105-172634', '0', 3000e3),
    'ALE-Pong': ('ALE43V0_EG_000_20201106-011948', '0', 5800e3),
}

