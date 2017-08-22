from vizdoom import *

class DoomWrapper:
	def __init__(self):
		game = DoomGame()
		game.load_config("/home/sean/projects/ViZDoom/scenarios/defend_the_center.cfg")
		game.set_window_visible(False)
		game.init()

		self.game = game

	def seed(seed):
		self.game.set_seed(seed)