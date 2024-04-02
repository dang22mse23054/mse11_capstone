module.exports = {

	TIME_ZONE: 'Asia/Tokyo',

	isEnabled: 1,

	BannedAPI: [
		'insertOrUpdate',
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

	FileTargets: {
		Temporary: 'tmp',
		Video: 'video',
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
		MP4: {
			ext: 'mp4',
			val: ['video/mp4']
		},
	}
};