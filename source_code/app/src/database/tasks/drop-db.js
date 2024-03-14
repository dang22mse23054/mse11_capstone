const dbConfig = require('../db-config');
const LogService = require('../../common/logger');
const log = LogService.getInstance(Number(process.env.LOG_DISPLAY_TYPE));

log.info('- Drop DB TASK -[BEGIN]-');
try {
	let dbName = process.env.DB_NAME;
	// connect without database selected
	var knex = require('knex')({ client: 'mysql', connection: dbConfig.basic});
	knex.raw(`DROP DATABASE IF EXISTS ${dbName}`).then((resp) =>{ 
		log.info('- Drop DB TASK -[END]-');
		knex.destroy();
	});
} catch (err) {
	log.error(err);
}
