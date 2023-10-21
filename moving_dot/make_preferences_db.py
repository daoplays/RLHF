import numpy as np
import random as random
from db import *
import os

def make_db(db_name = "preferences.db"):

	if (os.path.isfile(db_name)):
		log_db("database " + db_name + " already exists")
		return
	
	log_db("Creating database " + db_name)

	n_actions = 5
	actions = {}
	actions[0] = np.array([0,0])
	actions[1] = np.array([0,1])
	actions[2] = np.array([1,0])
	actions[3] = np.array([0,-1])
	actions[4] = np.array([-1,0])

	# for some reason the gym environment can only go between (4,4) and (205, 155)
	y_min = 4
	x_min = 4
	y_max = 205
	x_max = 155

	n_segments = 10000
	segments = []
	for i in range(n_segments):
		
		obs = np.array([np.random.randint(y_min, y_max+1), np.random.randint(x_min, x_max+1)])
		a = np.random.randint(0, n_actions)
		segments.append((obs, a))

	mid_point = np.array([105, 80])
	n_comparisons = 10000
	comparisons = []
	for i in range(n_comparisons):
		chosen_segments = random.sample(segments, 2)
		obs_1, action_1 = chosen_segments[0]
		obs_2, action_2 = chosen_segments[1]

		next_obs_1 = obs_1 + actions[action_1]
		next_obs_2 = obs_2 + actions[action_2]

		move_1 = np.sqrt(np.sum((obs_1 - mid_point)**2)) - np.sqrt(np.sum((next_obs_1 - mid_point)**2))
		move_2 = np.sqrt(np.sum((obs_2 - mid_point)**2)) - np.sqrt(np.sum((next_obs_2 - mid_point)**2))

		mu_1 = 0.5
		mu_2 = 0.5

		# move 1 has got closer to the center than move 2
		if (move_1 > move_2):
			mu_1 = 1
			mu_2 = 0
		
		# move 2 has got closer to the center than move 1
		if (move_2 > move_1):
			mu_1 = 0
			mu_2 = 1

		entry = [len(comparisons), int(obs_1[0]), int(obs_1[1]), int(action_1), int(obs_2[0]), int(obs_2[1]), int(action_2), mu_1, mu_2]
		comparisons.append(entry)

	conn = create_database_connection(db_name)
	insert_rows(conn, comparisons)
	conn.close()
