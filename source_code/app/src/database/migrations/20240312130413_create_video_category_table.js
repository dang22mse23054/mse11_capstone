exports.up = function (knex, Promise) {
	return knex.schema
		.dropTableIfExists('video_category')
		.createTable('video_category', function (table) {
			table.increments('id').primary();
			table.integer('videoId').unsigned().notNullable();
			table.integer('categoryId').unsigned().notNullable();

			table.unique(['videoId', 'categoryId']);
		});
};

exports.down = function (knex, Promise) {
	return knex.schema.dropTable('video_category');
};
