const ErrorCodes = {
	NOT_EXISTED_USER: { code: 10002, message: 'User Id is wrong' },
	DELETED_USER: { code: 10003, message: 'User Id is deleted' },
	OBSOLETE_DATA: { code: 10005, message: '別の人から変更しましたのでリロードしてください' },
	EXPRESS_VALIDATOR: {
		CODE: 20000,
	},
	INVALID_REQUEST: 20000,
	FILE: {
		UNSUPPORTED: { code: 11001, message: '非対応ファイル' },
		NO_PASSWORD: { code: 11002, message: 'クライアントに送付するレポート類(.zip .7z .xlsx)にはパスワード設定が必要です' },
		NAME_WITH_CLIENT_NAME: { code: 10006, message: 'ファイル名にクライアント名が含まれていません' },
	},
	PERMISSION_DENIED: { code: 403, message: 'Permission Denied' },
	UNKNOW_ERROR: { code: 500, message: 'Unknown Error' },

	GraphQL: {
		INVALID_OBJECT: 20001,
		PERMISSION_DENIED: 'GRAPHQL|PERMISSION_DENIED',
		UNKNOW_ERROR: 'GRAPHQL|UNKNOW_ERROR',
	},

	SCHEDULE: {
		SCHEDULE_NOT_EXISTED: { code: 30001, message: 'schedule is wrong' },
		SCHEDULE_FINISHED: { code: 30002, message: 'このスケジューラのタスクステータスは完了です' },
	},

	SLACK: {
		CHANNEL_NOT_FOUND: { code: 40001, message: 'チャンネルはありません' },
		RESTRICTED_ACTION: { code: 40002, message: 'このアクションは実行できません' },
		ALREADY_IN_CHANNEL: { code: 40003, message: 'メンバーはチャンネルに存在しています' },
	}
	
};

module.exports = {
	...ErrorCodes,
	Map: {
		[ErrorCodes.GraphQL.PERMISSION_DENIED]: ErrorCodes.PERMISSION_DENIED,
		[ErrorCodes.GraphQL.UNKNOW_ERROR]: ErrorCodes.UNKNOW_ERROR,
	}
};