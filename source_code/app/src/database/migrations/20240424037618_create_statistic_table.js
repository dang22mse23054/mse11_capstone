exports.up = function (knex) {
	return Promise.all([
		knex.schema
		.dropTableIfExists('statistic')
		.createTable('statistic', function (table) {
			table.integer('videoId').unsigned().notNullable();
			table.string('group', 50).notNullable();
			table.string('gender', 200).notNullable();
			table.string('age', 200).notNullable();
			table.string('happy', 500).notNullable();

			table.timestamp('createdAt').notNullable().defaultTo(knex.fn.now());
			table.timestamp('deletedAt').nullable();
			table.primary(['group', 'videoId', 'createdAt']);
		}),
	]);
};

exports.down = function (knex) {
	return Promise.all([
		knex.schema.dropTable('statistic'),
	]);
};
