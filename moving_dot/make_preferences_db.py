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
	x1_min = 4
	x2_min = 4
	x1_max = 205
	x2_max = 155

	x1 = 210
	x2 = 160
	mid_point = np.array([x1/2, x2/2])

	segments = []

	n_comparisons = 10000
	for i in range(n_comparisons):
		
		obs = np.array([np.random.randint(x1_min, x1_max+1), np.random.randint(x2_min, x2_max+1)])
		a = np.random.randint(0, n_actions)
		next_obs = obs + actions[a]
		segments.append((obs, next_obs, a))

		
	comparisons = []
	for i in range(n_comparisons):
		chosen_segments = random.sample(segments, 2)
		s1 = chosen_segments[0]
		s2 = chosen_segments[1]
		move_1 = np.sqrt(np.sum((s1[0]-mid_point)**2)) - np.sqrt(np.sum((s1[1]-mid_point)**2))
		move_2 = np.sqrt(np.sum((s2[0]-mid_point)**2)) - np.sqrt(np.sum((s2[1]-mid_point)**2))

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

		entry = [len(comparisons), int(s1[0][0]), int(s1[0][1]), int(s1[2]), int(s2[0][0]), int(s2[0][1]), int(s2[2]), mu_1, mu_2]
		comparisons.append(entry)

	conn = create_database_connection(db_name)
	insert_rows(conn, comparisons)
	conn.close()
