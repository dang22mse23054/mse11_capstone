module.exports = {

	TIME_ZONE: 'Asia/Tokyo',

	isEnabled: 1,

	BannedAPI: [
		'insertOrUpdateSchedule',
		'updateTask',
		// 'updateTaskProc'
	],

	RequestLimit: 50,

	JobTypes: {
		CsvReport: 1,
		Notification: 2,
	},

	JobNotiTypes: {
		Chatwork: 1,
		Slack: 2,
	},

	SystemRoles: {
		NORMAL: { label: '一般ユーザー', value: 1 },
		ADMIN: { label: 'システム管理者', value: 2 },
		MANAGER: { label: 'マネジャー', value: 3 },
	},

	ScheduleTypes: {
		SPOT: { id: 1, display: 'スポット' },
		REGULAR: { id: 2, display: '定例' },
	},

	DestTypes: {
		CLIENT: { id: 1, display: 'クライアント' },
		HANDLE_USER: { id: 2, display: '担当者' },
		DIVISION: { id: 3, display: '局' },
	},

	RequestChannel: {
		CHATWORK: { id: 1, display: 'Chatwork' },
		FORM: { id: 2, display: 'フォーム' },
		TASK_MANAGER: { id: 3, display: 'タスクマネージャー' },
		MAIL: { id: 4, display: 'メール' },
		SLACK: { id: 5, display: 'Slack' },
		BACKLOG: { id: 6, display: 'Backlog' },
		FACEBOOK: { id: 7, display: 'Facebook' },
	},

	FrequencyTypes: {
		WEEKLY: { id: 1, display: '週次' },
		MONTHLY_BY_WEEK: { id: 2, display: '月次(週)' },
		MONTHLY_BY_DATE: { id: 3, display: '月次(日)' },
	},
	FrequencyTypeIds: {
		WEEKLY: 1,
		MONTHLY_BY_WEEK: 2,
		MONTHLY_BY_DATE: 3,
	},

	DayOfWeek: [
		{ id: 1, en: 'Mon', jp: '月' },
		{ id: 2, en: 'Tue', jp: '火' },
		{ id: 3, en: 'Wed', jp: '水' },
		{ id: 4, en: 'Thu', jp: '木' },
		{ id: 5, en: 'Fri', jp: '金' },
		{ id: 6, en: 'Sat', jp: '土' },
		{ id: 0, en: 'Sun', jp: '日' },
	],

	DateFmt: {
		YYYYMMDD: 'YYYY-MM-DD',
	},

	ExportModes: {
		DeadlineDate: 1,
		FinishedDate: 2,
	},

	FileTargets: {
		Temporary: 'tmp',
		Schedule: 'schedule',
		Task: 'task',
		TaskProcess: 'task_proc',
	},

	FileTypes: {
		ZIP: {
			ext: 'zip',
			val: [
				// MacOS Zip
				'application/zip',
				// Windows Zip
				'application/x-zip-compressed',
			]
		},
		_7Z: {
			ext: '7z',
			val: [
				// Windows
				'application/octet-stream',
				// MacOS
				'application/x-7z-compressed'
			]
		},
		PDF: {
			ext: 'pdf',
			val: ['application/pdf']
		},
		CSV: {
			ext: 'csv',
			val: ['text/csv']
		},
		PNG: {
			ext: 'png',
			val: ['image/png']
		},
		JPEG: {
			ext: 'jpg',
			val: ['image/jpeg']
		},
		MS_OFFICE_FILE: {
			ext: null,
			val: ['application/x-tika-msoffice']
		},
		XLS: {
			ext: 'xls',
			val: ['application/vnd.ms-excel']
		},
		XLSX: {
			ext: 'xlsx',
			val: ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
		},
		XLSM: {
			ext: 'xlsm',
			val: [
				// MacOS
				'application/vnd.ms-excel.sheet.macroenabled.12',
				// Windows
				'application/vnd.ms-excel.sheet.macroEnabled.12'
			]
		},
		DOC: {
			ext: 'doc',
			val: ['application/msword']
		},
		DOCX: {
			ext: 'docx',
			val: ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']
		},
		PPT: {
			ext: 'ppt',
			val: ['application/vnd.ms-powerpoint']
		},
		PPTX: {
			ext: 'pptx',
			val: ['application/vnd.openxmlformats-officedocument.presentationml.presentation']
		},
	}
};