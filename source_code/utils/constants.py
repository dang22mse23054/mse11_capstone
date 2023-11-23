class Constants:
	__slots__ = ()

	# ======== CONFIG ======== #
	MUST_HAVE_ADMIN = True
	IS_USING_SLACK_ID = True

	# ======== Classes ======= #

	class Mode:
		__slots__ = ()
		DEMO = 'demo'
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