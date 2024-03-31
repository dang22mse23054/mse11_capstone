const ErrorCodes = {
	NOT_EXISTED_USER: { code: 10002, message: 'User Id is wrong' },
	DELETED_USER: { code: 10003, message: 'User Id is deleted' },
	OBSOLETE_DATA: { code: 10005, message: 'Data is out of date. Please try to reload' },
	EXPRESS_VALIDATOR: {
		CODE: 20000,
	},
	INVALID_REQUEST: 20000,
	FILE: {
		UNSUPPORTED: { code: 11001, message: 'Unsupported file' },
	},
	PERMISSION_DENIED: { code: 403, message: 'Permission Denied' },
	UNKNOW_ERROR: { code: 500, message: 'Unknown Error' },

	GraphQL: {
		INVALID_OBJECT: 20001,
		PERMISSION_DENIED: 'GRAPHQL|PERMISSION_DENIED',
		UNKNOW_ERROR: 'GRAPHQL|UNKNOW_ERROR',
	},
	
};

module.exports = {
	...ErrorCodes,
	Map: {
		[ErrorCodes.GraphQL.PERMISSION_DENIED]: ErrorCodes.PERMISSION_DENIED,
		[ErrorCodes.GraphQL.UNKNOW_ERROR]: ErrorCodes.UNKNOW_ERROR,
	}
};