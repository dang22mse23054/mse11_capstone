const callerId = require('caller-id');
const CustomLog = require('./custom-log');
const NEXT_PUBLIC_NODE_ENV = process.env.NEXT_PUBLIC_NODE_ENV || 'production';
const Sentry = require('../utils/Sentry');
const LOG_LEVEL = process.env.LOG_LEVEL || 'debug';
const LOG_DIR = 'logs';

const debugLogger = new CustomLog({
	logFolder: `${LOG_DIR}/debug`,
	suffixFileName: '-debug',
	logLevel: LOG_LEVEL
}).getLogger();

const infoLogger = new CustomLog({
	logFolder: `${LOG_DIR}/info`,
	suffixFileName: '-info',
	logLevel: LOG_LEVEL
}).getLogger();

const errorLogger = new CustomLog({
	logFolder: `${LOG_DIR}/error`,
	suffixFileName: '-error',
	logLevel: LOG_LEVEL
}).getLogger();

module.exports = {
	debug: (obj) => NEXT_PUBLIC_NODE_ENV !== 'production' ? debugLogger.debug(obj, callerId.getData()) : null,
	info: (obj) => infoLogger.info(obj, callerId.getData()),
	error: (obj, byPassSentry = false) => {
		errorLogger.error(obj, callerId.getData());

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
