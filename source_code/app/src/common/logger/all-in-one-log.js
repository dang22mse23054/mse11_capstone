const callerId = require('caller-id');
const CustomLog = require('./custom-log');
const NEXT_PUBLIC_NODE_ENV = process.env.NEXT_PUBLIC_NODE_ENV || 'production';
const Sentry = require('../utils/Sentry');
const LOG_LEVEL = process.env.LOG_LEVEL || 'info';
const LOG_DIR = 'logs';

const logger = new CustomLog({
	logFolder: `${LOG_DIR}/all`,
	suffixFileName: '-all',
	logLevel: LOG_LEVEL
}).getLogger();

module.exports = {
	debug: (obj) => NEXT_PUBLIC_NODE_ENV !== 'production' ? logger.debug(obj, callerId.getData()) : null,
	info: (obj) => logger.info(obj, callerId.getData()),
	error: (obj, byPassSentry = false) => {
		logger.error(obj, callerId.getData());

		if (byPassSentry != true) {
			let msg = 'Unknown Internal Error';
			if (typeof obj == 'string') {
				msg = obj;
			} else {
				msg = obj instanceof Error ? obj : obj.message || msg;
			}
			// Sentry.captureException(msg);
		}
	},
};