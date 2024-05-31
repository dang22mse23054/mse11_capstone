exports.up = function (knex) {
	return Promise.all([
		knex.schema
		.dropTableIfExists('log')
		.createTable('log', function (table) {
			table.string('id', 100).notNullable().primary();
			table.integer('videoId').unsigned().notNullable();
			table.string('gender', 200).notNullable();
			table.string('age', 200).notNullable();
			table.string('happy', 500).notNullable();

			table.timestamp('createdAt').notNullable().defaultTo(knex.fn.now());
			table.primary(['id', 'videoId', 'createdAt']);
		}),
	]);
};

exports.down = function (knex) {
	return Promise.all([
		knex.schema.dropTable('log'),
	]);
};
