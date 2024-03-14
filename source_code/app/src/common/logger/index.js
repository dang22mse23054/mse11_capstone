const Logger = {
	Type: {
		AllInOneLog: 1,
		SeparatedLog: 2
	},
	getInstance: (type = Number(process.env.LOG_DISPLAY_TYPE)) => {
		switch (type) {
			case Logger.Type.AllInOneLog: 
				const allInOneLogger = require('./all-in-one-log');
				return allInOneLogger;
			case Logger.Type.SeparatedLog: 
				const separatedLogger = require('./separated-log');
				return separatedLogger;
		}
	}
};

module.exports = Logger;