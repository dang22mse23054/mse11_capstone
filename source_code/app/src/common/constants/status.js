const ActionStatus = {
	SKIP: 0,
	NEW: 1,
	UPDATED: 2,
	DELETED: 3,
};

const StatusMap = { 1: ActionStatus.NEW, 2: ActionStatus.UPDATED, 3: ActionStatus.DELETED, 0: ActionStatus.SKIP };

const VideoStatus = {
	PAUSED: 0,
	PLAYING: 1,
	STOPPED: 2,
};

module.exports = {
	ActionStatus,
	VideoStatus,
	StatusMap
};