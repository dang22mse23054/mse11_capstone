const ActionStatus = {
	SKIP: 0,
	NEW: 1,
	UPDATED: 2,
	DELETED: 3,
};

const StatusMap = { 1: ActionStatus.NEW, 2: ActionStatus.UPDATED, 3: ActionStatus.DELETED, 0: ActionStatus.SKIP };

const ScheduleStatus = {
	RUNNING: 1,
	STOPED: 2,
	DELETED: 3
};

const TaskStatus = {
	OPEN: 0,
	SKIPED: 1,
	CLOSED: 2,
};

const TaskProcStatus = {
	OPEN: 0, // 未完了
	SKIPED: 1, // 対応なし
	CLOSED: 2 // 完了
};

const JobStatus = {
	PREPARE: 0,
	DOING: 1,
	SKIPED: 2,
	ERROR: 3,
	DONE: 4,
};

const ChannelMemberStatus = {
	ADD: 1,
	DELETE: 2,
};

module.exports = {
	ActionStatus,
	StatusMap,
	TaskStatus,
	TaskProcStatus,
	ScheduleStatus,
	JobStatus,
	ChannelMemberStatus
};