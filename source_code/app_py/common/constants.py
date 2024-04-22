class Constants:
	__slots__ = ()

	FACE_GROUPS =['Background', 'Face']

	COLOR = {
		"RED": (0, 0, 255),
		"BLUE": (255, 0, 0),
		"GREEN": (0, 255, 0),
		"YELLOW": (0, 255, 255),
		"PURPLE": (255, 0, 255),
	}
 
	# ======== Classes ======= #
	class Models():
		__slot__ = ()
		# Read only
		FACE_MODEL = 1
		AGE_GENDER_MODEL = 2
		EMOTION_MODEL = 3 
 
	# ======== Classes ======= #

	class Mode:
		__slots__ = ()
		DEMO = 'demo'
		VALDEMO = 'valdemo'
		TESTDEMO = 'testdemo'
		TRAIN = 'train'
		TEST = 'test'
		VALIDATE= 'valid'

	class DFColumns:
		__slots__ = ()
		PATH = 'path'
		NUM_BBOX = 'num_bbox'
		X1 = 'x1'
		Y1 = 'y1'
		W = 'w'
		H = 'h'
		BLUR = 'blur'
		EXPRESSION = 'expression'
		ILLUMINATION = 'illumination'
		INVALID = 'invalid'
		OCCLUSION = 'occlusion'
		POSE = 'pose'

	class Age:
		__slots__ = ()
		CHILDREN = 'children (0~12)'
		TEENAGERS = 'teenagers (13~17)'
		ADULT = 'adult (18~44)'
		MIDDLE_AGED = 'middle_aged (45~60)'
		ELDERLY = 'elderly (61~120)'

		Groups = {
      		# CHILDREN: 	(00~12) => 0
			CHILDREN: {'from': 0, 'to': 12}, 
			# TEENAGERS: 	(13~17) => 1
			TEENAGERS: {'from': 13, 'to': 17},
			# ADULT: 		(18~44) => 2
			ADULT: {'from': 18, 'to': 44}, 
			# MIDDLE_AGED:(45~60) => 3
			MIDDLE_AGED: {'from': 45, 'to': 60},
			# ELDERLY: 	(61~12) => 4
			ELDERLY: {'from': 61, 'to': 120},
		}

	class Emotion:
		__slots__ = ()
		ANGRY = 'angry'
		DISGUST = 'disgust'
		FEAR = 'fear'
		SAD = 'sad'
		NEUTRAL = 'neutral'
		HAPPY = 'happy'
		SURPRISE = 'surprise'

		Groups = [
			ANGRY,
			DISGUST,
			FEAR,
			SAD,
			NEUTRAL,
			HAPPY,
			SURPRISE,
		]

		Augmentor = {
			ANGRY: 3000,
			DISGUST: 6500,
			FEAR: 3000,
			SAD: 2000,
			NEUTRAL: 2000,
			HAPPY: 0,
			SURPRISE: 3500,
		}

	class Position:
		__slots__ = ()
		LEFT = 1
		RIGHT = 2
		CENTER = 0
		UP = 3
		DOWN = 4

		Label = {
			UP: 'UP',
			DOWN: 'DOWN',
			CENTER: 'CENTER',
			LEFT: 'LEFT',
			RIGHT: 'RIGHT',
		}
