const { ErrorCodes } = require('commonDir/constants');
const ErrorObject = require('apiDir/dto/ErrorObject');

const Permission = {

	hasRole: (acceptedRoles) => next => async (root, args, context, info) => {
		const { userInfo } = context;
		if (acceptedRoles.includes(userInfo.roleId)) {
			return next(root, args, context, info);
		}

		throw new ErrorObject(ErrorCodes.GraphQL.PERMISSION_DENIED);
	}
};

module.exports = Permission;