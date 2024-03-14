const dbConfig = require('../db-config');
const LogService = require('../../common/logger');
const log = LogService.getInstance(Number(process.env.LOG_DISPLAY_TYPE));

log.info('- Init DB TASK -[BEGIN]-');
try {
	let dbName = process.env.DB_NAME;
	// connect without database selected
	var knex = require('knex')({ client: 'mysql', connection: dbConfig.basic});
	knex.raw(`CREATE DATABASE IF NOT EXISTS ${dbName} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci `).then((resp) =>{ 
		log.info('- Init DB TASK -[END]-');
		knex.destroy();
	});
} catch (error) {
	log.error(error);
}