// Note: dotenv load .env begin from root folder
// const result = require('dotenv').config({ path: '.env' });
require('dotenv').config({ path: '.env' });

// Knex configuration object.
// Note: need to set NEXT_PUBLIC_NODE_ENV value (.env file)
const BasicConfig = {
	host: process.env.DB_HOST,
	port: process.env.DB_PORT,
	user: process.env.DB_USERNAME,
	password: process.env.DB_PASSWORD,
	acquireConnectionTimeout: 60000,
	timezone: 'UTC',
	charset: 'utf8mb4',
};

const DbConfig = {
	basic: { ...BasicConfig },
	readonly: {
		client: 'mysql',
		connection: process.env.DB_CONNECTION_STR || {
			...BasicConfig, 
			host: process.env.DB_RO_HOST,
			port: process.env.DB_RO_PORT,
			database: process.env.DB_NAME
		},
		pool: { 
			min: parseInt(process.env.DB_POOL_MIN), 
			max: parseInt(process.env.DB_POOL_MAX),
			idleTimeoutMillis: parseInt(process.env.DB_POOL_IDLE),
		},
		debug: Boolean(process.env.DB_DEBUG == 'true') || false,
	},
	master: {
		// log: {
		//     warn(message) {
		//     },
		//     error(message) {
		//     },
		//     deprecate(message) {
		//     },
		//     debug(message) {
		//     },
		// },
		client: 'mysql',
		connection: process.env.DB_CONNECTION_STR || {
			...BasicConfig, 
			database: process.env.DB_NAME
		},
		pool: { 
			min: parseInt(process.env.DB_POOL_MIN), 
			max: parseInt(process.env.DB_POOL_MAX),
			idleTimeoutMillis: parseInt(process.env.DB_POOL_IDLE),
		},
		debug: Boolean(process.env.DB_DEBUG == 'true') || false,
		migrations: {
			/* The table name used for storing the migration state */
			tableName: 'knex_migrations',
			/* Directory containing the migration files. Default ./migrations */
			directory: './src/database/migrations'
		},
		seeds: {
			/* Default records */
			directory: './src/database/seeds'
		}
	},
};

module.exports = DbConfig;