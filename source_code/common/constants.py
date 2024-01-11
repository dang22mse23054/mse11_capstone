class Constants:
	__slots__ = ()

	FACE_GROUPS =['Background', 'Face']
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
			CHILDREN: {'from': 0, 'to': 12}, 
			TEENAGERS: {'from': 13, 'to': 17},
			ADULT: {'from': 18, 'to': 44}, 
			MIDDLE_AGED: {'from': 45, 'to': 60},
			ELDERLY: {'from': 61, 'to': 120},
		}