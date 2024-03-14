const winston = require('winston');
require('winston-daily-rotate-file');
const { createLogger, format, transports } = winston;
const { combine, timestamp, /* label,  */printf, colorize } = format;

const fs = require('fs');
const path = require('path');
const callerId = require('caller-id');

const LOG_LEVEL = process.env.LOG_LEVEL || 'debug';
const SHOW_CONSOLE_LOG = Boolean(process.env.SHOW_CONSOLE_LOG == 'true') || false;
const LOG_DIR = 'logs';

const Formators = {
	reformatMsg: () => printf((info) => {
		let caller = info.caller;
		// let alignFormat = format.align();
		// info = alignFormat.transform(info);

		const messasge = typeof(info.message) === 'string' ? info.message : `\n\r${JSON.stringify(info.message, null, 4)}`;

		if (caller) {
			return `[${info.timestamp}] [${info.level}] [${path.relative('.', caller.filePath)}:${caller.lineNumber}] [${caller.functionName}] ${messasge}`;
		}

		return `[${info.timestamp}] [${info.level}] ${info.message}`;
	}),

	errorStackFormat: format((info) => {
		// console.log(({}).toString.call(info).match(/\s([a-zA-Z]+)/)[1].toLowerCase())
		// console.log(JSON.stringify(info))

		if (info.level == 'error') {
			let msg = info.message.stack || info.message;

			if (info.message.getStack instanceof Function) {
				msg = info.message.getStack();
			}

			info = { ...info, message: msg };
		}
		return info;
	})
};

module.exports = class CustomLog {
	constructor({ logFolder = null, suffixFileName = '-result', logLevel = LOG_LEVEL, exitOnError = false }) {
		this.logFolder = logFolder || LOG_DIR;
		this.suffixFileName = suffixFileName;
		this.logLevel = logLevel;
		this.exitOnError = exitOnError;
		this.getLogger = this.getLogger.bind(this);
		this.initTransports = this.initTransports.bind(this);
	}


	getLogger() {
		// Make log directory
		try {
			if (!fs.existsSync(this.logFolder)) { fs.mkdirSync(this.logFolder, { recursive: true }); }
		} catch (err) {
			// Here you get the error when the file was not found,
			// but you also get any other error
		}

		const logger = createLogger({
			level: this.logLevel,
			format: this.initFormators(),
			transports: this.initTransports(),
			exceptionHandlers: [new transports.File({ filename: `${LOG_DIR}/exceptions.log`, handleExceptions: true }), new transports.Console()],
			rejectionHandlers: [new transports.File({ filename: `${LOG_DIR}/exceptions.log`, handleExceptions: true }), new transports.Console()],
			exitOnError: this.exitOnError
		});

		return {
			debug: function (obj, caller = callerId.getData()) {
				logger.debug(obj, { caller });
			},
			error: function (obj, caller = callerId.getData()) {
				logger.error(obj, { caller });
			},
			info: function (obj, caller = callerId.getData()) {
				logger.info(obj, { caller });
			},
		};
	}

	getConsoleLogger({ logLevel = LOG_LEVEL, exitOnError = false }) {
		return createLogger({
			level: logLevel,
			format: combine(
				timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
				colorize({
					colors: {
						warn: 'yellow',
						debug: 'blue',
						info: 'green',
						error: 'red'
					}
				}),
				Formators.reformatMsg(),
			),
			transports: [new transports.Console()],
			exitOnError: exitOnError
		});
	}

	initTransports() {
		let arr = [];
		let dailyRotationTransport = new (winston.transports.DailyRotateFile)({
			dirname: this.logFolder,
			filename: `%DATE%${this.suffixFileName}.log`,
			datePattern: 'YYYYMMDD-HH',
			zippedArchive: true,
			maxSize: '1g',
			maxFiles: '30d'
		});
		arr.push(dailyRotationTransport);

		if (SHOW_CONSOLE_LOG) {
			arr.push(this.getConsoleLogger({ logLevel: this.logLevel, exitOnError: this.exitOnError }));
		}

		return arr;
	}

	initFormators() {
		return combine(
			Formators.errorStackFormat(),
			timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
			Formators.reformatMsg(),
		);
	}

};