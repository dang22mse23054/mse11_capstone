const ErrorCodes = require('./error-codes');
const LoginTypes = require('./login-type');
const RoutePaths = require('./route-paths');
const Common = require('./common');
const Status = require('./status');
const Sorting = require('./sorting');

module.exports = {
	ErrorCodes,
	LoginTypes,
	RoutePaths,
	Common,
	Sorting,
	...Status,
};